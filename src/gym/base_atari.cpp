//
// Created by dewe on 3/27/21.
//

#include "cppdrl/gym/base_atari.h"
#include "cppdrl/gym/spaces/box.h"
#include "cppdrl/gym/spaces/discrete.h"
#include <boost/random/random_device.hpp>
#include <utility>
#include "cppdrl/misc/helper.h"
#include <boost/random.hpp>
#include <boost/functional/hash.hpp>
#include <algorithm>    // std::find




vector<unsigned char>  BaseAtari::get_screen(bool gray_screen)
{
    vector<unsigned char> screen;
    if(gray_screen)
        ale->getScreenGrayscale(screen);
    else
        ale->getScreenRGB(screen);


    return screen;
}

torch::Tensor BaseAtari::reset()
{
    ale->reset_game();
    return getObs();
}

StepResult BaseAtari::step(const variant<torch::Tensor, map<std::string, torch::Tensor>> &action)
{
    ale::Action game_action = action_set[std::get<torch::Tensor>(action).item<int>()];
    float reward = 0.0;
    int frame_skip_ = frame_skip[0];
    if (frame_skip.size() > 1)
    {
        boost::mt19937 gen;
        boost::uniform_int dist(frame_skip[0], frame_skip[1]);
        frame_skip_ = dist(gen);
    }

    while(frame_skip_ >= 0)
    {
        reward += ale->act(game_action);
        frame_skip_--;
    }

    bool done = ale->game_over();

    return {getObs(),
            torch::tensor(reward),
            torch::tensor(done),
            {{"ale_lives", float(ale->lives())}}};
}

BaseAtari::BaseAtari(string const& _game, std::optional<ale::game_mode_t> const& mode,
                     std::optional<ale::difficulty_t> const& game_difficulty, bool image_obs,
                     vector<int> const &frame_skip, float repeat_Action_prob,
                     bool full_action_space):GymEnv(), game(_game), game_mode(mode),
                     game_difficulty(game_difficulty),
                     image_obs(image_obs),
                     frame_skip(frame_skip),
                     ale(std::make_unique<ale::ALEInterface>(false)) {

    ale->setFloat("repeat_action_probability", repeat_Action_prob);

    game_path += "/" + game + ".bin";
#ifdef __USE_SDL
    ale->setBool("display_screen", false);
    ale->setBool("sound", false);
#endif

    seed(0);




    action_set = {full_action_space ? ale->getLegalActionSet() : ale->getMinimalActionSet()};
    action_space = std::make_unique<Discrete>(action_set.size());

    if (image_obs)
        observation_space = std::make_unique<Box>(0.f, 255.f, vector<int64_t>{static_cast<int64_t>(ale->getScreen().width()),
                                                               static_cast<int64_t>(ale->getScreen().height())},
                                                  torch::kInt);
    else
        observation_space = std::make_unique<Box>(0.f, 255.f, vector<int64_t>{RAM_LENGTH}, torch::kInt);


}

array<uint64_t, 2> BaseAtari::seed(optional<size_t> const& value) {

    size_t seed1;
    if (value.has_value()) {
        seed1 = value.value();
    }
    else{
        seed1 = rand();
    }

    size_t seed2 = seed1;
    torch::manual_seed(seed1);
    boost::hash_combine(seed2, 1);
    ale->setInt("random_seed", seed2);

    try {
        ale->loadROM(game_path);
        if(game_mode.has_value()) {
            auto game_modes = ale->getAvailableModes();
            auto found = std::find(begin(game_modes), end(game_modes), game_mode.value());

            assert( found.base() && "Invalid Gamemode");
            ale->setMode(game_mode.value());
        }
        if(game_mode.has_value())
        {
            auto game_difficulties = ale->getAvailableDifficulties();
            auto found = std::find(begin(game_difficulties), end(game_difficulties), game_difficulty.value());
            assert( found.base() && "Invalid GameDifficulty");
            ale->setDifficulty(game_mode.value());
        }

    } catch (std::exception const &exp)
    {
        std::cout << exp.what() << std::endl;
    }

    return {seed1, seed2};

}

vector<unsigned char> BaseAtari::getRam()
{
    auto ram = ale->getRAM();
    vector<unsigned char> buffer(RAM_LENGTH);
    memcpy(buffer.data(), ram.array(), ram.size());
    return buffer;
}

vector<unsigned char> BaseAtari::getImage()    {
    vector<unsigned char> buffer;
    ale->getScreenRGB(buffer);
    return buffer;
}

torch::Tensor BaseAtari::getObs()
{
    if(image_obs)
    {
        auto buffer = getImage();
        auto screen = ale->getScreen();
        auto tensor_img = torch::from_blob(buffer.data(), {static_cast<long>(screen.height()),
                                                           static_cast<long>(screen.width())}, torch::kUInt8);
        return tensor_img;
    }
    else
    {
        auto tensor_ram = torch::from_blob(getRam().data(), {RAM_LENGTH}, torch::kUInt8);
        return tensor_ram;
    }
}

std::optional<cv::Mat> BaseAtari::render(bool human)     {
    auto image = get_screen(false);
    auto screen = ale->getScreen();
    auto _image = cv::Mat(screen.height(), screen.width(), CV_8UC3, image.data());
    if(!human)
        return _image;


    return std::nullopt;
}

