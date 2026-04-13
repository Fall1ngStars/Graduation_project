// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_HPP_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_HPP_

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
// Member 'states'
#include "ranger_msgs/msg/detail/actuator_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__ranger_msgs__msg__ActuatorStateArray __attribute__((deprecated))
#else
# define DEPRECATED__ranger_msgs__msg__ActuatorStateArray __declspec(deprecated)
#endif

namespace ranger_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ActuatorStateArray_
{
  using Type = ActuatorStateArray_<ContainerAllocator>;

  explicit ActuatorStateArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit ActuatorStateArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _states_type =
    std::vector<ranger_msgs::msg::ActuatorState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ranger_msgs::msg::ActuatorState_<ContainerAllocator>>>;
  _states_type states;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__states(
    const std::vector<ranger_msgs::msg::ActuatorState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<ranger_msgs::msg::ActuatorState_<ContainerAllocator>>> & _arg)
  {
    this->states = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ranger_msgs__msg__ActuatorStateArray
    std::shared_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ranger_msgs__msg__ActuatorStateArray
    std::shared_ptr<ranger_msgs::msg::ActuatorStateArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ActuatorStateArray_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->states != other.states) {
      return false;
    }
    return true;
  }
  bool operator!=(const ActuatorStateArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ActuatorStateArray_

// alias to use template instance with default allocator
using ActuatorStateArray =
  ranger_msgs::msg::ActuatorStateArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ranger_msgs

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_HPP_
