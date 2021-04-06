//
// Created by dewe on 3/28/21.
//

#pragma once
#include "cppdrl/policies/policy.h"
#include "cppdrl/misc/strategy.h"

using std::begin;
using std::endl;
using std::cout;
using std::end;


template<typename ModelType, typename OptimizerType, typename OptimizerOption, typename StrategyType>
class PG : public Policy<ModelType, OptimizerType, OptimizerOption> {

protected:
    unique_ptr<Strategy> strategy;
    unique_ptr<NNBase> target_model;

public:
    explicit PG(TrainConfig const& train_config, std::optional<OptimizerOption> const& optim_option):
    Policy<ModelType, OptimizerType, OptimizerOption>(train_config, optim_option),
        strategy(std::make_unique<StrategyType>(train_config)),
        target_model(std::make_unique<ModelType>(train_config))
    {
        target_model->to(train_config.device);
        update_target();
    }

    ModelOutputDict act(SampleBatch const& sample_batch, torch::Tensor const& hidden_states) override
    {
        return strategy->act(this->model.get(), sample_batch, hidden_states);
    }

    LossDict optimize(SampleBatch& batch, const torch::Tensor& hidden_states, TrainConfig const& train_config) override
    {
        auto sampleBatch = this->sample(batch, train_config.sgd_mini_batch, train_config);
        auto old_obs = sampleBatch[SampleBatchKey::OBS]; // make copy
        auto online_q = this->model->forward(sampleBatch, hidden_states);

        sampleBatch[SampleBatchKey::OBS] = sampleBatch[SampleBatchKey::NEW_OBS];
        auto target_q = target_model->forward(sampleBatch, hidden_states);

        torch::Tensor max_q = std::get<0>(torch::max(target_q[ModelOutputKey::LOGITS].detach(), 1)).unsqueeze(1); // max q_values

        //todo: make gamma outside
        float gamma  = 1.00;
        auto td_target = sampleBatch[SampleBatchKey::REWARDS] +
                (gamma * max_q * (1 - sampleBatch[SampleBatchKey::DONES]));

        auto max_values = online_q[ModelOutputKey::LOGITS].gather( 1,sampleBatch[SampleBatchKey::ACTIONS]);
        torch::Tensor loss = (max_values - td_target).pow(2).mul(0.5).mean();
        this->optimizer->zero_grad();
        loss.backward();
//        torch::nn::utils::clip_grad_norm_(this->model->Parameters(), maxGradientNorm);
        this->optimizer->step();

        if (torch::isnan(loss).template item<bool>())
        {
            cout << sampleBatch[SampleBatchKey::ACTIONS] << endl;
            cout << sampleBatch[SampleBatchKey::NEW_OBS] << endl;
            cout << sampleBatch[SampleBatchKey::OBS] << endl;
            cout << sampleBatch[SampleBatchKey::REWARDS] << endl;
            cout << sampleBatch[SampleBatchKey::DONES] << endl;
        }


        // todo: add grad norm
        // todo: add learning rate decay

        return {{LossDictKey::POLICY_LOSS, loss.item<float>()}};
    }

    void update_target()
    {
        for(int i = 0; i < target_model->parameters().size(); i++)
        {
            target_model->parameters()[i].data().copy_(this->model->parameters()[i].data());
        }

    }

};
