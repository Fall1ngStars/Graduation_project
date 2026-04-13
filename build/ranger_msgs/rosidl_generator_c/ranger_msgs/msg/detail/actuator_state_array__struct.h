// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'states'
#include "ranger_msgs/msg/detail/actuator_state__struct.h"

/// Struct defined in msg/ActuatorStateArray in the package ranger_msgs.
typedef struct ranger_msgs__msg__ActuatorStateArray
{
  std_msgs__msg__Header header;
  ranger_msgs__msg__ActuatorState__Sequence states;
} ranger_msgs__msg__ActuatorStateArray;

// Struct for a sequence of ranger_msgs__msg__ActuatorStateArray.
typedef struct ranger_msgs__msg__ActuatorStateArray__Sequence
{
  ranger_msgs__msg__ActuatorStateArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__ActuatorStateArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE_ARRAY__STRUCT_H_
