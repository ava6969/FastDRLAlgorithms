//
// Created by dewe on 3/28/21.
//

#pragma once

#include "torch/torch.h"
#include "memory"
#include "cppdrl/models/nn_base.h"
#include <cppdrl/misc/helper.h>

using std::unique_ptr;

template<typename ModelType, typename OptimizerType, typename OptimizerOption>
class Policy
{
/**
 * Policy is in charge of samplying and optimizing model
 * */
protected:
    unique_ptr<NNBase> model;
    unique_ptr<torch::optim::Optimizer>  optimizer{};

public:
    explicit Policy(TrainConfig const& trainConfig, std::optional<OptimizerOption> const& optim_option):
    model(std::make_unique<ModelType>(trainConfig))
    {
        model->to(trainConfig.device); // only dqn
        if (optim_option.has_value())
        {
            optimizer =  std::make_unique<OptimizerType>(model->parameters(), optim_option.value());

        }
    }

    NNBase* get_model() const {return model.get(); }
    virtual ModelOutputDict act(SampleBatch const& sample_batch, torch::Tensor const& hidden_state) = 0;

    virtual LossDict optimize(SampleBatch& train_batch, const torch::Tensor& hidden_state, TrainConfig const& train_config) = 0;

    virtual SampleBatch sample(SampleBatch const& batch, size_t batch_size, TrainConfig const& trainConfig)
    {
        auto t_size = batch.at(SampleBatchKey::OBS).size(0);
        SampleBatch out;
        auto shuffled_indices = torch::slice(torch::randperm(t_size, TensorOptions().dtype(kLong).device(trainConfig.device)), 0, 0, batch_size);
        out[SampleBatchKey::OBS] = batch.at(SampleBatchKey::OBS).to(trainConfig.device).index_select(0, shuffled_indices).toType(torch::kF32);
        out[SampleBatchKey::ACTIONS] = batch.at(SampleBatchKey::ACTIONS).to(trainConfig.device).index_select(0, shuffled_indices).toType(torch::kLong);
        out[SampleBatchKey::NEW_OBS] = batch.at(SampleBatchKey::NEW_OBS).to(trainConfig.device).index_select(0, shuffled_indices).toType(torch::kF32);
        out[SampleBatchKey::REWARDS] = batch.at(SampleBatchKey::REWARDS).to(trainConfig.device).index_select(0,shuffled_indices).toType(torch::kF32);
        out[SampleBatchKey::DONES] = batch.at(SampleBatchKey::DONES).to(trainConfig.device).index_select(0, shuffled_indices).toType(torch::kFloat32);
        return out;
    }

    virtual void update_target() = 0;

    void update_weight(NNBase* _model)
    {
        torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        auto new_params = model->named_parameters(); // implement this
        auto params = _model->named_parameters(true /*recurse*/);
        auto buffers = _model->named_buffers(true /*recurse*/);
        for (auto& val : new_params) {
            auto name = val.key();
            auto* t = params.find(name);
            if (t != nullptr) {
                t->copy_(val.value());
            } else {
                t = buffers.find(name);
                if (t != nullptr) {
                    t->copy_(val.value());
                }
            }
        }
        torch::autograd::GradMode::set_enabled(true);
    }

    virtual ~Policy()=default;
};

