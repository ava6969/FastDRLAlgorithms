//
// Created by dewe on 3/28/21.
//

#ifndef CPPDRL_DQN_TRAINER_H
#define CPPDRL_DQN_TRAINER_H

#include "trainer.h"
#include "memory"
#include "boost/circular_buffer.hpp"
#include "type_traits"

using std::unique_ptr;


template<typename ModelType, typename OptimizerType, typename OptimizerOption, typename EnvType, typename StrategyType>
class DQNTrainer : public Trainer<ModelType, OptimizerType, OptimizerOption, EnvType>{

protected:
    SampleBatch replay_queue;
public:
    explicit DQNTrainer(TrainConfig const& train_config,
                        OptimizerOption const& optim_option): Trainer<ModelType, OptimizerType, OptimizerOption, EnvType>(train_config)
    {
        auto obs_shape = train_config.observation_space->getShape();
        auto obs_type = train_config.observation_space->getType();
        auto action_shape = train_config.action_space->getShape();
        auto action_type = train_config.action_space->getType();

        this->trainer = std::make_unique<DQN<ModelType, OptimizerType, OptimizerOption, StrategyType>>(train_config, optim_option);
        auto n_workers = train_config.num_workers;

        auto new_obs_shape = concat({long(n_workers), long(train_config.buffer_size)}, obs_shape);
        replay_queue[SampleBatchKey::OBS] = torch::empty(new_obs_shape, torch::TensorOptions(obs_type));

        auto new_action_shape = concat({long(n_workers), long(train_config.buffer_size)}, 1);
        replay_queue[SampleBatchKey::ACTIONS] = torch::empty(new_action_shape, torch::TensorOptions(action_type));

        replay_queue[SampleBatchKey::NEXT_OBS] = torch::empty(new_obs_shape, torch::TensorOptions(obs_type));

        replay_queue[SampleBatchKey::REWARDS] = torch::empty({long(n_workers), long(train_config.buffer_size), 1},
                                                       torch::TensorOptions(torch::kFloat32));

        replay_queue[SampleBatchKey::DONES] = torch::empty({long(n_workers), long(train_config.buffer_size), 1},
                                                           torch::TensorOptions(torch::kF32));

        replay_queue[SampleBatchKey::T] = torch::zeros({long(n_workers), 1}, torch::TensorOptions(torch::kI64));
    }

    void train_async()
    {
        size_t total_timesteps = 0;
        size_t iteration = 0;


        this->start();
        this->awake();

        while(total_timesteps < this->train_config.total_timesteps)
        {
            torch::Tensor time_step_sum = this->time_step_tensor.sum();
            total_timesteps = time_step_sum.item<long>();

            //todo: relpay start > buffer size for now
            if(total_timesteps > this->train_config.sgd_mini_batch && total_timesteps > this->train_config.replay_start_size)
            {
                if(total_timesteps % this->train_config.update_freq == 0)
                {
                    auto flattened_batch = getTrainBatch();

                    auto loss = this->trainer->optimize(flattened_batch, {}, this->train_config);
                    this->transport_weights();
                    if(iteration % this->train_config.max_target_update == 0) // every c update - update target
                    {
                        dynamic_cast<DQN<ModelType, OptimizerType, OptimizerOption, StrategyType>*>(this->trainer.get())->update_target();
                    }
                    iteration++;

                    // todo: create logging and debugging system
                    auto mean_reward = torch::mean(this->last_100_reward);
                    auto std_reward = torch::std(this->last_100_reward);
                    if (iteration % this->train_config.report_every == 0)
                        cout << "iteration: " << iteration << "reward: mean " << mean_reward << " std " << std_reward << endl;
                }

            }
        }
        this->stop();
    }

    void train() override {


        /// intialize

        size_t total_timesteps = 0;
        size_t iteration = 0;
        EnvConfig env_config = this->train_config.env_config;
        int max_episode = 1000000;
        int index = 0;
        size_t r_idx = 0;
        int current_episode = 0;
        env_config["worker_id"] = index +  1;

        this->worker_envs[index] = std::make_unique<EnvType>(env_config);
        this->worker_policies[index] = this->create_policy();

        while (current_episode < max_episode) {

            // get initial states
            torch::Tensor recurrent_state;
            auto observation = this->worker_envs[index]->reset();
            StepResult result;

            /// start loop
            while (true) {
                total_timesteps =  replay_queue[SampleBatchKey::T][index].template item<int>();
                std::tie(observation, recurrent_state, result) = step(observation, recurrent_state, index, total_timesteps);

                this->last_100_reward[index][r_idx] += result.rewards;

                if (total_timesteps > this->train_config.replay_start_size &&
                        (total_timesteps % this->train_config.update_freq) == 0) {

                    auto flattened_batch = this->getTrainBatch();
                    auto loss = this->trainer->optimize(flattened_batch, {}, this->train_config);

                    if (iteration % this->train_config.report_every == 0)
                    {

                        // todo: create logging and debugging system
                        torch::Tensor mean_reward = torch::mean(this->last_100_reward);
                        torch::Tensor std_reward = torch::std(this->last_100_reward);

                        cout << "epsiodes: "<<  current_episode << " total timesteps: " << total_timesteps << " iteration: " << iteration << " reward: mean " <<
                             mean_reward.item<float>() << " std " <<
                             std_reward.item<float>() << " loss: " << loss[LossDictKey::POLICY_LOSS] << endl;

                        cout << this->last_100_reward << endl;
                    }
                    iteration++;
                }

                if (total_timesteps % this->train_config.max_target_update == 0) {
                    this->trainer->update_target();
                }

                replay_queue[SampleBatchKey::T][index]+=1;
                torch::Tensor all_done = result.done;
                if (all_done.item<bool>()) {
                    break;
                }

            }
            current_episode++;
            r_idx = current_episode % LAST_100_EPISODIC_REW;
            this->last_100_reward[index][r_idx] = 0;

        }
    }

    SampleBatch getTrainBatch() override
    {
        SampleBatch sampleBatch;
        torch::Tensor min_idx_ = torch::minimum(replay_queue[SampleBatchKey::T].min(), torch::tensor(int(this->train_config.buffer_size)));

        int idx = min_idx_.template item<int>();
        std::for_each(begin(this->train_config.sample_keys), end(this->train_config.sample_keys),[&](auto& key) mutable {
                    auto tensor_ = replay_queue[key].slice(1, 0, idx);
                    tensor_ = tensor_.flatten(0, 1);
                    sampleBatch[key] = tensor_;
        });
        return sampleBatch;
    }

    void run(int index) override
    {
        this->worker_states[index] = WorkerState::IDLE;
        // wait to be set to running
        std::unique_lock<std::mutex> lck(this->mutexes[index]);
        cout << "worker " << index << " : is waiting to be started" << endl;
        while (this->worker_states[index] == WorkerState::IDLE) this->cv_s[index].wait(lck);

        cout << "worker " << index << " : is running" << endl;
        torch::Tensor recurrent_state;
        auto observation = this->worker_envs[index]->reset();
        size_t t = 0;

        replay_queue[SampleBatchKey::OBS][index][t]  = observation;
        size_t current_episode = 0;
        while(this->worker_states[index] != WorkerState::STOPPED)
        {
            SampleBatch temp_batch;
            this->time_step_tensor[index] += 1; // keep incrementing timesteps

            // todo: how much sample do you want -> history
            temp_batch[SampleBatchKey::OBS] = observation.to(this->train_config.device);
            auto model_output = this->worker_policies[index]->act(temp_batch, recurrent_state) ; // todo: rearrange how this is sent

            recurrent_state = model_output[SampleBatchKey::HIDDEN_STATE];

            auto result = this->worker_envs[index]->step(model_output[SampleBatchKey::ACTIONS]);

            replay_queue[SampleBatchKey::ACTIONS][index][t] = model_output[SampleBatchKey::ACTIONS];
            replay_queue[SampleBatchKey::NEXT_OBS][index][t] = result.observation;
            replay_queue[SampleBatchKey::REWARDS][index][t] = result.rewards;
            replay_queue[SampleBatchKey::DONES][index][t] = result.done;


            auto r_idx = current_episode % 100;
            this->last_100_reward[index][r_idx] += result.rewards;
            t++;

            if (t % this->train_config.buffer_size == 0)
                t = 0;

            // todo: add infos
            torch::Tensor all_done = result.done;
            if(all_done.item<bool>())
            {
                current_episode++;
                r_idx = current_episode % 100;
                this->last_100_reward[index][r_idx] = 0;

                replay_queue[SampleBatchKey::OBS][index][t] = this->worker_envs[index]->reset();
            } else
                replay_queue[SampleBatchKey::OBS][index][t] = result.observation;
        }
        cout << "worker " << index << " : is stopped" << endl;
    }

    unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>> create_policy() override
    {
        return std::make_unique<DQN<ModelType, OptimizerType, OptimizerOption, StrategyType>>(this->train_config, std::nullopt);
    }

    std::tuple<torch::Tensor, torch::Tensor, StepResult> step(
            torch::Tensor const& observation,
            torch::Tensor const& recurrent_state,
            int index,
            int total_ts)
    {
        /// create worker batch
        SampleBatch worker_batch;
        /// get action
        worker_batch[SampleBatchKey::OBS] = observation.to(this->train_config.device);
        auto model_output = this->worker_policies[index]->act(worker_batch, recurrent_state);
        torch::Tensor _recurrent_state = model_output[SampleBatchKey::HIDDEN_STATE]; // todo: when we start rnn make sure to push to device

        /// step in environment
        StepResult result = this->worker_envs[index]->step(model_output[SampleBatchKey::ACTIONS].cpu());

        /// update queue

        size_t t = total_ts % this->train_config.buffer_size;

        replay_queue[SampleBatchKey::OBS][index][t] = observation;
        replay_queue[SampleBatchKey::ACTIONS][index][t] = model_output[SampleBatchKey::ACTIONS].cpu();
        replay_queue[SampleBatchKey::NEXT_OBS][index][t] = result.observation;
        replay_queue[SampleBatchKey::REWARDS][index][t] = result.rewards;
        replay_queue[SampleBatchKey::DONES][index][t] = result.done;

        return {result.observation, model_output[SampleBatchKey::HIDDEN_STATE], result };
    }


};


#endif //CPPDRL_DQN_TRAINER_H
