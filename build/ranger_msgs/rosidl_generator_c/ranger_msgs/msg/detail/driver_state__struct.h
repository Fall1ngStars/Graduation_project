// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/DriverState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/DriverState in the package ranger_msgs.
typedef struct ranger_msgs__msg__DriverState
{
  float driver_voltage;
  float driver_temperature;
  float motor_temperature;
  uint32_t driver_state;
} ranger_msgs__msg__DriverState;

// Struct for a sequence of ranger_msgs__msg__DriverState.
typedef struct ranger_msgs__msg__DriverState__Sequence
{
  ranger_msgs__msg__DriverState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__DriverState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__DRIVER_STATE__STRUCT_H_
