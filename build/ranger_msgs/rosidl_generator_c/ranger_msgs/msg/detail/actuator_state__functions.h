// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice

#ifndef RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__FUNCTIONS_H_
#define RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "ranger_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "ranger_msgs/msg/detail/actuator_state__struct.h"

/// Initialize msg/ActuatorState message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * ranger_msgs__msg__ActuatorState
 * )) before or use
 * ranger_msgs__msg__ActuatorState__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__init(ranger_msgs__msg__ActuatorState * msg);

/// Finalize msg/ActuatorState message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
void
ranger_msgs__msg__ActuatorState__fini(ranger_msgs__msg__ActuatorState * msg);

/// Create msg/ActuatorState message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * ranger_msgs__msg__ActuatorState__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
ranger_msgs__msg__ActuatorState *
ranger_msgs__msg__ActuatorState__create();

/// Destroy msg/ActuatorState message.
/**
 * It calls
 * ranger_msgs__msg__ActuatorState__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
void
ranger_msgs__msg__ActuatorState__destroy(ranger_msgs__msg__ActuatorState * msg);

/// Check for msg/ActuatorState message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__are_equal(const ranger_msgs__msg__ActuatorState * lhs, const ranger_msgs__msg__ActuatorState * rhs);

/// Copy a msg/ActuatorState message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__copy(
  const ranger_msgs__msg__ActuatorState * input,
  ranger_msgs__msg__ActuatorState * output);

/// Initialize array of msg/ActuatorState messages.
/**
 * It allocates the memory for the number of elements and calls
 * ranger_msgs__msg__ActuatorState__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__Sequence__init(ranger_msgs__msg__ActuatorState__Sequence * array, size_t size);

/// Finalize array of msg/ActuatorState messages.
/**
 * It calls
 * ranger_msgs__msg__ActuatorState__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
void
ranger_msgs__msg__ActuatorState__Sequence__fini(ranger_msgs__msg__ActuatorState__Sequence * array);

/// Create array of msg/ActuatorState messages.
/**
 * It allocates the memory for the array and calls
 * ranger_msgs__msg__ActuatorState__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
ranger_msgs__msg__ActuatorState__Sequence *
ranger_msgs__msg__ActuatorState__Sequence__create(size_t size);

/// Destroy array of msg/ActuatorState messages.
/**
 * It calls
 * ranger_msgs__msg__ActuatorState__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
void
ranger_msgs__msg__ActuatorState__Sequence__destroy(ranger_msgs__msg__ActuatorState__Sequence * array);

/// Check for msg/ActuatorState message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__Sequence__are_equal(const ranger_msgs__msg__ActuatorState__Sequence * lhs, const ranger_msgs__msg__ActuatorState__Sequence * rhs);

/// Copy an array of msg/ActuatorState messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_ranger_msgs
bool
ranger_msgs__msg__ActuatorState__Sequence__copy(
  const ranger_msgs__msg__ActuatorState__Sequence * input,
  ranger_msgs__msg__ActuatorState__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // RANGER_MSGS__MSG__DETAIL__ACTUATOR_STATE__FUNCTIONS_H_
