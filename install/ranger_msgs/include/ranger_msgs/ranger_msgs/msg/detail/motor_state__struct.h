// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/MotorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/MotorState in the package ranger_msgs.
typedef struct ranger_msgs__msg__MotorState
{
  int32_t rpm;
  float current;
  int32_t pulse_count;
  float motor_angles;
  float motor_speeds;
} ranger_msgs__msg__MotorState;

// Struct for a sequence of ranger_msgs__msg__MotorState.
typedef struct ranger_msgs__msg__MotorState__Sequence
{
  ranger_msgs__msg__MotorState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__MotorState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__MOTOR_STATE__STRUCT_H_
