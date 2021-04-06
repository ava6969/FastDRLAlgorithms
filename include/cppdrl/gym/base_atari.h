//
// Created by dewe on 3/27/21.
//

#pragma once

#include <cppdrl/gym/gym_env.h>
#include "vector"
#include "array"
#include <ale_interface.hpp>
#include "optional"
#include "string"


using std::optional;
using std::unique_ptr;
using std::vector;
using std::array;
using std::string;

#ifdef __USE_SDL
#include <SDL.h>
#endif




using std::map;

//torch::Tensor to_ram(ale::ALEInterface* ale)
//{
//    auto ram = ale->getRAM();
//    return torch::from_blob(&(ram.array()), {ram.size()});
//}

class BaseAtari : public GymEnv {

private:
    std::unique_ptr<ale::ALEInterface> ale;
    ale::ActionVect action_set;
    uint64_t lives{};
    bool game_over{};
    std::optional<ale::game_mode_t> game_mode;
    std::optional<ale::difficulty_t> game_difficulty;
    std::string game;
    bool image_obs;
    vector<int> frame_skip = {};
    string game_path = "/home/dewe/DRLAlgorithms/roms";

public:
    BaseAtari(std::string const& game,
              std::optional<ale::game_mode_t> const &mode,
              std::optional<ale::difficulty_t> const &game_difficulty,
              bool image_obs = true,
              vector<int> const &frame_skip = {},
              float repeat_Action_prob = 0,
              bool full_action_space = false);

    vector<unsigned char> get_screen(bool gray_screen);

    int get_lives() const { return ale->lives();}

    [[nodiscard]] ale::ALEInterface *ALE() const { return ale.get(); }

    torch::Tensor reset() override;

    StepResult step(std::variant<torch::Tensor, map<std::string, torch::Tensor>> const &action) override;

    array<uint64_t, 2> seed(optional<size_t> const& value);

    vector<unsigned char> getRam();

    vector<unsigned char>  getImage();

    torch::Tensor getObs();

    std::optional<cv::Mat> render(bool human) override;


};