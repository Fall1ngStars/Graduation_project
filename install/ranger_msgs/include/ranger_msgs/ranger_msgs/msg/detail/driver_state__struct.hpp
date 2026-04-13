// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ranger_msgs:msg/DriverState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_HPP_
#define RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ranger_msgs__msg__DriverState __attribute__((deprecated))
#else
# define DEPRECATED__ranger_msgs__msg__DriverState __declspec(deprecated)
#endif

namespace ranger_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DriverState_
{
  using Type = DriverState_<ContainerAllocator>;

  explicit DriverState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->driver_voltage = 0.0f;
      this->driver_temperature = 0.0f;
      this->motor_temperature = 0.0f;
      this->driver_state = 0ul;
    }
  }

  explicit DriverState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->driver_voltage = 0.0f;
      this->driver_temperature = 0.0f;
      this->motor_temperature = 0.0f;
      this->driver_state = 0ul;
    }
  }

  // field types and members
  using _driver_voltage_type =
    float;
  _driver_voltage_type driver_voltage;
  using _driver_temperature_type =
    float;
  _driver_temperature_type driver_temperature;
  using _motor_temperature_type =
    float;
  _motor_temperature_type motor_temperature;
  using _driver_state_type =
    uint32_t;
  _driver_state_type driver_state;

  // setters for named parameter idiom
  Type & set__driver_voltage(
    const float & _arg)
  {
    this->driver_voltage = _arg;
    return *this;
  }
  Type & set__driver_temperature(
    const float & _arg)
  {
    this->driver_temperature = _arg;
    return *this;
  }
  Type & set__motor_temperature(
    const float & _arg)
  {
    this->motor_temperature = _arg;
    return *this;
  }
  Type & set__driver_state(
    const uint32_t & _arg)
  {
    this->driver_state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ranger_msgs::msg::DriverState_<ContainerAllocator> *;
  using ConstRawPtr =
    const ranger_msgs::msg::DriverState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::DriverState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::DriverState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ranger_msgs__msg__DriverState
    std::shared_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ranger_msgs__msg__DriverState
    std::shared_ptr<ranger_msgs::msg::DriverState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DriverState_ & other) const
  {
    if (this->driver_voltage != other.driver_voltage) {
      return false;
    }
    if (this->driver_temperature != other.driver_temperature) {
      return false;
    }
    if (this->motor_temperature != other.motor_temperature) {
      return false;
    }
    if (this->driver_state != other.driver_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const DriverState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DriverState_

// alias to use template instance with default allocator
using DriverState =
  ranger_msgs::msg::DriverState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_HPP_
