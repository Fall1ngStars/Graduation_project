// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ranger_msgs/msg/detail/actuator_state__rosidl_typesupport_introspection_c.h"
#include "ranger_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ranger_msgs/msg/detail/actuator_state__functions.h"
#include "ranger_msgs/msg/detail/actuator_state__struct.h"


// Include directives for member types
// Member `motor`
#include "ranger_msgs/msg/motor_state.h"
// Member `motor`
#include "ranger_msgs/msg/detail/motor_state__rosidl_typesupport_introspection_c.h"
// Member `driver`
#include "ranger_msgs/msg/driver_state.h"
// Member `driver`
#include "ranger_msgs/msg/detail/driver_state__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ranger_msgs__msg__ActuatorState__init(message_memory);
}

void ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_fini_function(void * message_memory)
{
  ranger_msgs__msg__ActuatorState__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_member_array[3] = {
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__ActuatorState, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "motor",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__ActuatorState, motor),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "driver",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ranger_msgs__msg__ActuatorState, driver),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_members = {
  "ranger_msgs__msg",  // message namespace
  "ActuatorState",  // message name
  3,  // number of fields
  sizeof(ranger_msgs__msg__ActuatorState),
  ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_member_array,  // message members
  ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_init_function,  // function to initialize message memory (memory has to be allocated)
  ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_type_support_handle = {
  0,
  &ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ranger_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, ActuatorState)() {
  ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, MotorState)();
  ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ranger_msgs, msg, DriverState)();
  if (!ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_type_support_handle.typesupport_identifier) {
    ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ranger_msgs__msg__ActuatorState__rosidl_typesupport_introspection_c__ActuatorState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
