//
// Created by dewe on 3/30/21.
//

#ifndef CPPDRL_PG_TRAINER_H
#define CPPDRL_PG_TRAINER_H

#include "cppdrl/trainers/trainer.h"
#include "cppdrl/models/nn_base.h"
#include "cppdrl/policies/policy.h"

using std::unique_ptr;
template<typename ModelType,
        typename OptimizerType,
        typename OptimizerOption,
        typename EnvType,
        typename PolicyType>
class PGTrainer : public Trainer<ModelType, OptimizerType, OptimizerOption, EnvType>{

private:
    SampleBatch buffer;
    uint32_t max_fragment_size;

public:
    explicit PGTrainer(TrainConfig& train_config,
              OptimizerOption const& optim_option): Trainer<ModelType, OptimizerType, OptimizerOption, EnvType>(train_config)
    {
        // assert train_batch is multiple of (rollout frag length * n_workers)
        int rollout_fragment_length = train_config.rollout_length;
        int n_workers = train_config.num_workers;
        int buffer_size = train_config.buffer_size;


        int multiple = int(buffer_size / (n_workers*rollout_fragment_length));

        auto obs_shape = train_config.observation_space->getShape();
        auto obs_type = train_config.observation_space->getType();
        auto action_shape = train_config.action_space->getShape();
        auto action_type = train_config.action_space->getType();

        // worker, transition_n, rollout_fragment, Data
        auto new_obs_shape = concat({long(n_workers), multiple, long(rollout_fragment_length)}, obs_shape);
        buffer[SampleBatchKey::OBS] = torch::empty(new_obs_shape, torch::TensorOptions(obs_type));

        auto new_action_shape = concat({long(n_workers), multiple, long(rollout_fragment_length)}, 1);
        buffer[SampleBatchKey::ACTIONS] = torch::empty(new_action_shape, torch::TensorOptions(action_type);

        buffer[SampleBatchKey::NEXT_OBS] = torch::empty(new_obs_shape, torch::TensorOptions(obs_type));

        buffer[SampleBatchKey::REWARDS] = torch::empty({long(n_workers), multiple, long(rollout_fragment_length), 1},
                                                       torch::TensorOptions(torch::kFloat32));

        buffer[SampleBatchKey::LOGP] = torch::empty({long(n_workers), multiple, long(rollout_fragment_length), 1},
                                                       torch::TensorOptions(torch::kFloat32));

        buffer[SampleBatchKey::ENTROPY] = torch::empty({long(n_workers), multiple, long(rollout_fragment_length), 1},
                                                       torch::TensorOptions(torch::kFloat32));

        buffer[SampleBatchKey::DONES] = torch::empty({long(n_workers), multiple, long(rollout_fragment_length), 1},
                                                     torch::TensorOptions(torch::kBool));

        buffer[SampleBatchKey::RETURNS] = torch::empty({long(n_workers), multiple, long(rollout_fragment_length), 1},
                                                     torch::TensorOptions(torch::kBool));

        buffer[SampleBatchKey::T] = torch::zeros({long(n_workers), 1}, torch::TensorOptions(torch::kI32));

        max_fragment_size =  multiple;
    }

    SampleBatch getTrainBatch() override
    {
        SampleBatch sampleBatch;
        std::for_each(begin(this->train_config.sample_keys), end(this->train_config.sample_keys),[&](auto& key) mutable {
            auto tensor_ = sampleBatch[key].flatten(0, 2);
            sampleBatch[key] = tensor_;
        });
        return sampleBatch;
    }
    void run(int index) override
    {
        while(this->worker_states[index] != WorkerState::STOPPED)
        {
            this->worker_states[index] = WorkerState::IDLE;
            // wait to be set to running
            std::unique_lock<std::mutex> lck(this->mutexes[index]);
            cout << "worker " << index << " : is waiting to be started" << endl;
            while (this->worker_states[index] == WorkerState::IDLE) this->cv_s[index].wait(lck);

            int frag_batch=0;
            torch::Tensor recurrent_state;
            buffer[SampleBatchKey::OBS][index] = this->worker_envs[index]->reset().to(this->train_config.device);
            SampleBatch temp_batch;

            cout << "worker " << index << " : is running" << endl;
            for(int i = 0; i < max_fragment_size; i++)
            {
                int t = 0;
                while(t < this->train_config.rollout_length)
                {

                    buffer[SampleBatchKey::T][index] += 1; // keep incrementing timesteps

                    // todo: how much sample do you want -> history
                    temp_batch[SampleBatchKey::OBS] = buffer[SampleBatchKey::OBS][index][frag_batch][t].to(this->train_config.device);
                    ModelOutputDict model_output;
                    {
                        torch::NoGradGuard guard;
                        model_output = this->worker_policies[index]->act(temp_batch, recurrent_state);
                    }

                    recurrent_state = model_output[SampleBatchKey::HIDDEN_STATE];
                    auto result = this->worker_envs[index]->step(model_output[SampleBatchKey::ACTIONS]);

                    buffer[SampleBatchKey::NEXT_OBS][index][frag_batch][t] = result.observation;
                    buffer[SampleBatchKey::REWARDS][index][frag_batch][t] = result.rewards;
                    buffer[SampleBatchKey::DONES][index][t] = result.done;
                    buffer[SampleBatchKey::LOGP][index][frag_batch][t] = model_output[SampleBatchKey::LOGP];
                    buffer[SampleBatchKey::ENTROPY][index][frag_batch][t] = model_output[SampleBatchKey::ENTROPY];
                    buffer[SampleBatchKey::VF_PREDS][index][frag_batch][t] = model_output[SampleBatchKey::VF_PREDS];
                    // todo: add infos


                    if(! result.done.item<bool>())
                    {
                        buffer[SampleBatchKey::OBS][index][frag_batch][t] = this->worker_envs[index]->reset();
                    } else
                        buffer[SampleBatchKey::OBS][index][frag_batch][t] = result.observation;

                    t++;
                    {
                        torch::NoGradGuard guard;
                        auto next_value = this->worker_policies[index]->predict(result.observation,
                                                                      recurrent_state).detach();
                        this->compute_return(index, frag_batch, next_value);
                    }

                }
                frag_batch++;
            }

            this->worker_states[index] = WorkerState::IDLE;
            cout << "worker " << index << " : is done running" << endl;

        }
    }

    virtual void compute_return(int index,
                                int fragment_batch,
                                torch::Tensor const& next_value)
    {
        // update returns
        this->buffer[SampleBatchKey::RETURNS][index][fragment_batch][-1] = next_value;
        // todo returns of zeros will be added wasted
        for (int step = this->train_config.frag_length; step >= 0; --step)
        {
            this->rewards[step] = (this->buffer[SampleBatchKey::RETURNS][index][fragment_batch] * gamma *  this->buffer[SampleBatchKey::DONES][index][fragment_batch][step + 1] +
                    this->buffer[SampleBatchKey::REWARDS][index][fragment_batch][step]);
        }
    }

    virtual void train()=0;

    unique_ptr<Policy<ModelType, OptimizerType, OptimizerOption>> create_policy() override
    {
        return std::make_unique<PolicyType<ModelType, OptimizerType, OptimizerOption>>(this->train_config, std::nullopt);
    }


};

#endif //CPPDRL_PG_TRAINER_H
