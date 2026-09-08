#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"
#include "plantIK.h"

namespace {

constexpr int kArmDof = 14;
constexpr int kPoseValues = 7;  // xyz + xyzw
constexpr int kInputValues = kArmDof + 2 * kPoseValues;

std::vector<double> parse_line(const std::string& line) {
  std::istringstream stream(line);
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value) {
    values.push_back(value);
  }
  return values;
}

Eigen::Quaterniond quaternion_xyzw(const std::vector<double>& values, int offset) {
  const Eigen::Quaterniond q(
      values[offset + 6], values[offset + 3], values[offset + 4], values[offset + 5]);
  const double norm = q.norm();
  if (!(norm > 1e-12) || !std::isfinite(norm)) {
    throw std::runtime_error("invalid quaternion");
  }
  return q.normalized();
}

void print_failure() {
  std::cout << "0\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: kuavo_plantik_server <biped_v3_arm.urdf>\n";
    return 2;
  }

  try {
    drake::systems::DiagramBuilder<double> builder;
    auto* plant = builder.AddSystem<drake::multibody::MultibodyPlant<double>>(0.001);
    drake::multibody::Parser parser(plant);
    parser.AddModels(std::filesystem::path(argv[1]));
    plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("torso"));
    plant->Finalize();

    if (plant->num_positions() != kArmDof) {
      throw std::runtime_error("expected a 14-DoF arm-only Drake URDF");
    }

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();
    auto& plant_context = diagram->GetMutableSubsystemContext(*plant, context.get());
    (void)plant_context;

    HighlyDynamic::CoMIK solver(
        plant, {"torso", "l_hand_roll", "r_hand_roll"});
    HighlyDynamic::IKParams params;
    params.pos_cost_weight = 0.0;
    params.constraint_mode = 0;

    std::cout << std::setprecision(17);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }
      try {
        const auto values = parse_line(line);
        if (values.size() != kInputValues) {
          throw std::runtime_error("expected 28 numeric values");
        }
        for (const double value : values) {
          if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite input");
          }
        }

        Eigen::VectorXd q0(kArmDof);
        for (int i = 0; i < kArmDof; ++i) {
          q0[i] = values[i];
        }

        const int left_offset = kArmDof;
        const int right_offset = kArmDof + kPoseValues;
        Eigen::Vector3d left_position(
            values[left_offset], values[left_offset + 1], values[left_offset + 2]);
        Eigen::Vector3d right_position(
            values[right_offset], values[right_offset + 1], values[right_offset + 2]);
        HighlyDynamic::FramePoseVec poses{
            {Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()},
            {quaternion_xyzw(values, left_offset), left_position},
            {quaternion_xyzw(values, right_offset), right_position},
        };

        Eigen::VectorXd solution;
        if (!solver.solve(poses, q0, solution, params) || solution.size() != kArmDof) {
          print_failure();
          continue;
        }

        std::cout << "1";
        for (int i = 0; i < kArmDof; ++i) {
          std::cout << ' ' << solution[i];
        }
        std::cout << '\n' << std::flush;
      } catch (const std::exception& error) {
        std::cerr << "request failed: " << error.what() << '\n';
        print_failure();
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "initialization failed: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
