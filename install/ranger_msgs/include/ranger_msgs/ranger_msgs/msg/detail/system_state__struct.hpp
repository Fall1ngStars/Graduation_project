// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ranger_msgs:msg/SystemState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_HPP_
#define RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ranger_msgs__msg__SystemState __attribute__((deprecated))
#else
# define DEPRECATED__ranger_msgs__msg__SystemState __declspec(deprecated)
#endif

namespace ranger_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SystemState_
{
  using Type = SystemState_<ContainerAllocator>;

  explicit SystemState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vehicle_state = 0;
      this->control_mode = 0;
      this->error_code = 0;
      this->battery_voltage = 0.0;
      this->motion_mode = 0;
    }
  }

  explicit SystemState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->vehicle_state = 0;
      this->control_mode = 0;
      this->error_code = 0;
      this->battery_voltage = 0.0;
      this->motion_mode = 0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _vehicle_state_type =
    uint8_t;
  _vehicle_state_type vehicle_state;
  using _control_mode_type =
    uint8_t;
  _control_mode_type control_mode;
  using _error_code_type =
    uint16_t;
  _error_code_type error_code;
  using _battery_voltage_type =
    double;
  _battery_voltage_type battery_voltage;
  using _motion_mode_type =
    uint8_t;
  _motion_mode_type motion_mode;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__vehicle_state(
    const uint8_t & _arg)
  {
    this->vehicle_state = _arg;
    return *this;
  }
  Type & set__control_mode(
    const uint8_t & _arg)
  {
    this->control_mode = _arg;
    return *this;
  }
  Type & set__error_code(
    const uint16_t & _arg)
  {
    this->error_code = _arg;
    return *this;
  }
  Type & set__battery_voltage(
    const double & _arg)
  {
    this->battery_voltage = _arg;
    return *this;
  }
  Type & set__motion_mode(
    const uint8_t & _arg)
  {
    this->motion_mode = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint8_t VEHICLE_STATE_NORMAL =
    0u;
  static constexpr uint8_t VEHICLE_STATE_ESTOP =
    1u;
  static constexpr uint8_t VEHICLE_STATE_EXCEPTION =
    2u;
  static constexpr uint8_t CONTROL_MODE_RC =
    0u;
  static constexpr uint8_t CONTROL_MODE_CAN =
    1u;
  static constexpr uint8_t MOTION_MODE_DUAL_ACKERMAN =
    0u;
  static constexpr uint8_t MOTION_MODE_PARALLEL =
    1u;
  static constexpr uint8_t MOTION_MODE_SPINNING =
    2u;
  static constexpr uint8_t MOTION_MODE_SIDE_SLIP =
    3u;

  // pointer types
  using RawPtr =
    ranger_msgs::msg::SystemState_<ContainerAllocator> *;
  using ConstRawPtr =
    const ranger_msgs::msg::SystemState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::SystemState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::SystemState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ranger_msgs__msg__SystemState
    std::shared_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ranger_msgs__msg__SystemState
    std::shared_ptr<ranger_msgs::msg::SystemState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SystemState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->vehicle_state != other.vehicle_state) {
      return false;
    }
    if (this->control_mode != other.control_mode) {
      return false;
    }
    if (this->error_code != other.error_code) {
      return false;
    }
    if (this->battery_voltage != other.battery_voltage) {
      return false;
    }
    if (this->motion_mode != other.motion_mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const SystemState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SystemState_

// alias to use template instance with default allocator
using SystemState =
  ranger_msgs::msg::SystemState_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::VEHICLE_STATE_NORMAL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::VEHICLE_STATE_ESTOP;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::VEHICLE_STATE_EXCEPTION;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::CONTROL_MODE_RC;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::CONTROL_MODE_CAN;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::MOTION_MODE_DUAL_ACKERMAN;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::MOTION_MODE_PARALLEL;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::MOTION_MODE_SPINNING;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t SystemState_<ContainerAllocator>::MOTION_MODE_SIDE_SLIP;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__SYSTEM_STATE__STRUCT_HPP_
