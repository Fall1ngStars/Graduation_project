// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ranger_msgs:msg/MotorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_HPP_
#define RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ranger_msgs__msg__MotorState __attribute__((deprecated))
#else
# define DEPRECATED__ranger_msgs__msg__MotorState __declspec(deprecated)
#endif

namespace ranger_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MotorState_
{
  using Type = MotorState_<ContainerAllocator>;

  explicit MotorState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->rpm = 0l;
      this->current = 0.0f;
      this->pulse_count = 0l;
      this->motor_angles = 0.0f;
      this->motor_speeds = 0.0f;
    }
  }

  explicit MotorState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->rpm = 0l;
      this->current = 0.0f;
      this->pulse_count = 0l;
      this->motor_angles = 0.0f;
      this->motor_speeds = 0.0f;
    }
  }

  // field types and members
  using _rpm_type =
    int32_t;
  _rpm_type rpm;
  using _current_type =
    float;
  _current_type current;
  using _pulse_count_type =
    int32_t;
  _pulse_count_type pulse_count;
  using _motor_angles_type =
    float;
  _motor_angles_type motor_angles;
  using _motor_speeds_type =
    float;
  _motor_speeds_type motor_speeds;

  // setters for named parameter idiom
  Type & set__rpm(
    const int32_t & _arg)
  {
    this->rpm = _arg;
    return *this;
  }
  Type & set__current(
    const float & _arg)
  {
    this->current = _arg;
    return *this;
  }
  Type & set__pulse_count(
    const int32_t & _arg)
  {
    this->pulse_count = _arg;
    return *this;
  }
  Type & set__motor_angles(
    const float & _arg)
  {
    this->motor_angles = _arg;
    return *this;
  }
  Type & set__motor_speeds(
    const float & _arg)
  {
    this->motor_speeds = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ranger_msgs::msg::MotorState_<ContainerAllocator> *;
  using ConstRawPtr =
    const ranger_msgs::msg::MotorState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::MotorState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::MotorState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ranger_msgs__msg__MotorState
    std::shared_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ranger_msgs__msg__MotorState
    std::shared_ptr<ranger_msgs::msg::MotorState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MotorState_ & other) const
  {
    if (this->rpm != other.rpm) {
      return false;
    }
    if (this->current != other.current) {
      return false;
    }
    if (this->pulse_count != other.pulse_count) {
      return false;
    }
    if (this->motor_angles != other.motor_angles) {
      return false;
    }
    if (this->motor_speeds != other.motor_speeds) {
      return false;
    }
    return true;
  }
  bool operator!=(const MotorState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MotorState_

// alias to use template instance with default allocator
using MotorState =
  ranger_msgs::msg::MotorState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_HPP_
