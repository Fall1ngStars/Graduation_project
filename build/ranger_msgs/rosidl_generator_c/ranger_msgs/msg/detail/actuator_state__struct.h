// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'motor'
#include "ranger_msgs/msg/detail/motor_state__struct.h"
// Member 'driver'
#include "ranger_msgs/msg/detail/driver_state__struct.h"

/// Struct defined in msg/ActuatorState in the package ranger_msgs.
typedef struct ranger_msgs__msg__ActuatorState
{
  uint32_t id;
  ranger_msgs__msg__MotorState motor;
  ranger_msgs__msg__DriverState driver;
} ranger_msgs__msg__ActuatorState;

// Struct for a sequence of ranger_msgs__msg__ActuatorState.
typedef struct ranger_msgs__msg__ActuatorState__Sequence
{
  ranger_msgs__msg__ActuatorState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__ActuatorState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__STRUCT_H_
