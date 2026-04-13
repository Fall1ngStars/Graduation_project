// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'motor'
#include "ranger_msgs/msg/detail/motor_state__struct.hpp"
// Member 'driver'
#include "ranger_msgs/msg/detail/driver_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ranger_msgs__msg__ActuatorState __attribute__((deprecated))
#else
# define DEPRECATED__ranger_msgs__msg__ActuatorState __declspec(deprecated)
#endif

namespace ranger_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ActuatorState_
{
  using Type = ActuatorState_<ContainerAllocator>;

  explicit ActuatorState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : motor(_init),
    driver(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0ul;
    }
  }

  explicit ActuatorState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : motor(_alloc, _init),
    driver(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0ul;
    }
  }

  // field types and members
  using _id_type =
    uint32_t;
  _id_type id;
  using _motor_type =
    ranger_msgs::msg::MotorState_<ContainerAllocator>;
  _motor_type motor;
  using _driver_type =
    ranger_msgs::msg::DriverState_<ContainerAllocator>;
  _driver_type driver;

  // setters for named parameter idiom
  Type & set__id(
    const uint32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__motor(
    const ranger_msgs::msg::MotorState_<ContainerAllocator> & _arg)
  {
    this->motor = _arg;
    return *this;
  }
  Type & set__driver(
    const ranger_msgs::msg::DriverState_<ContainerAllocator> & _arg)
  {
    this->driver = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ranger_msgs::msg::ActuatorState_<ContainerAllocator> *;
  using ConstRawPtr =
    const ranger_msgs::msg::ActuatorState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::ActuatorState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::ActuatorState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ranger_msgs__msg__ActuatorState
    std::shared_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ranger_msgs__msg__ActuatorState
    std::shared_ptr<ranger_msgs::msg::ActuatorState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ActuatorState_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->motor != other.motor) {
      return false;
    }
    if (this->driver != other.driver) {
      return false;
    }
    return true;
  }
  bool operator!=(const ActuatorState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ActuatorState_

// alias to use template instance with default allocator
using ActuatorState =
  ranger_msgs::msg::ActuatorState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_HPP_
