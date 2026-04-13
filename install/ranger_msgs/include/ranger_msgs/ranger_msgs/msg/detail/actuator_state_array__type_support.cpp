// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "ranger_msgs/msg/detail/actuator_state_array__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace ranger_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ActuatorStateArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) ranger_msgs::msg::ActuatorStateArray(_init);
}

void ActuatorStateArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<ranger_msgs::msg::ActuatorStateArray *>(message_memory);
  typed_message->~ActuatorStateArray();
}

size_t size_function__ActuatorStateArray__states(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<ranger_msgs::msg::ActuatorState> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ActuatorStateArray__states(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<ranger_msgs::msg::ActuatorState> *>(untyped_member);
  return &member[index];
}

void * get_function__ActuatorStateArray__states(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<ranger_msgs::msg::ActuatorState> *>(untyped_member);
  return &member[index];
}

void fetch_function__ActuatorStateArray__states(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const ranger_msgs::msg::ActuatorState *>(
    get_const_function__ActuatorStateArray__states(untyped_member, index));
  auto & value = *reinterpret_cast<ranger_msgs::msg::ActuatorState *>(untyped_value);
  value = item;
}

void assign_function__ActuatorStateArray__states(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<ranger_msgs::msg::ActuatorState *>(
    get_function__ActuatorStateArray__states(untyped_member, index));
  const auto & value = *reinterpret_cast<const ranger_msgs::msg::ActuatorState *>(untyped_value);
  item = value;
}

void resize_function__ActuatorStateArray__states(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<ranger_msgs::msg::ActuatorState> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ActuatorStateArray_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs::msg::ActuatorStateArray, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "states",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<ranger_msgs::msg::ActuatorState>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs::msg::ActuatorStateArray, states),  // bytes offset in struct
    nullptr,  // default value
    size_function__ActuatorStateArray__states,  // size() function pointer
    get_const_function__ActuatorStateArray__states,  // get_const(index) function pointer
    get_function__ActuatorStateArray__states,  // get(index) function pointer
    fetch_function__ActuatorStateArray__states,  // fetch(index, &value) function pointer
    assign_function__ActuatorStateArray__states,  // assign(index, value) function pointer
    resize_function__ActuatorStateArray__states  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ActuatorStateArray_message_members = {
  "ranger_msgs::msg",  // message namespace
  "ActuatorStateArray",  // message name
  2,  // number of fields
  sizeof(ranger_msgs::msg::ActuatorStateArray),
  ActuatorStateArray_message_member_array,  // message members
  ActuatorStateArray_init_function,  // function to initialize message memory (memory has to be allocated)
  ActuatorStateArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ActuatorStateArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ActuatorStateArray_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace ranger_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<ranger_msgs::msg::ActuatorStateArray>()
{
  return &::ranger_msgs::msg::rosidl_typesupport_introspection_cpp::ActuatorStateArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, ranger_msgs, msg, ActuatorStateArray)() {
  return &::ranger_msgs::msg::rosidl_typesupport_introspection_cpp::ActuatorStateArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
