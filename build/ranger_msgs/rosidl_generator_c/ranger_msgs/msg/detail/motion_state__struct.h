// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from ranger_msgs:msg/MotionState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__MOTION_STATE__STRUCT_H_
#define RANGER_MSGS__MSG__DETAIL__MOTION_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'MOTION_MODE_DUAL_ACKERMAN'.
enum
{
  ranger_msgs__msg__MotionState__MOTION_MODE_DUAL_ACKERMAN = 0
};

/// Constant 'MOTION_MODE_PARALLEL'.
enum
{
  ranger_msgs__msg__MotionState__MOTION_MODE_PARALLEL = 1
};

/// Constant 'MOTION_MODE_SPINNING'.
enum
{
  ranger_msgs__msg__MotionState__MOTION_MODE_SPINNING = 2
};

/// Constant 'MOTION_MODE_SIDE_SLIP'.
enum
{
  ranger_msgs__msg__MotionState__MOTION_MODE_SIDE_SLIP = 3
};

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/MotionState in the package ranger_msgs.
typedef struct ranger_msgs__msg__MotionState
{
  std_msgs__msg__Header header;
  /// motion state
  uint8_t motion_mode;
} ranger_msgs__msg__MotionState;

// Struct for a sequence of ranger_msgs__msg__MotionState.
typedef struct ranger_msgs__msg__MotionState__Sequence
{
  ranger_msgs__msg__MotionState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} ranger_msgs__msg__MotionState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__MOTION_STATE__STRUCT_H_
