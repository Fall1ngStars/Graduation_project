// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ranger_msgs:msg/ActuatorState.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/actuator_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `motor`
#include "ranger_msgs/msg/detail/motor_state__functions.h"
// Member `driver`
#include "ranger_msgs/msg/detail/driver_state__functions.h"

bool
ranger_msgs__msg__ActuatorState__init(ranger_msgs__msg__ActuatorState * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // motor
  if (!ranger_msgs__msg__MotorState__init(&msg->motor)) {
    ranger_msgs__msg__ActuatorState__fini(msg);
    return false;
  }
  // driver
  if (!ranger_msgs__msg__DriverState__init(&msg->driver)) {
    ranger_msgs__msg__ActuatorState__fini(msg);
    return false;
  }
  return true;
}

void
ranger_msgs__msg__ActuatorState__fini(ranger_msgs__msg__ActuatorState * msg)
{
  if (!msg) {
    return;
  }
  // id
  // motor
  ranger_msgs__msg__MotorState__fini(&msg->motor);
  // driver
  ranger_msgs__msg__DriverState__fini(&msg->driver);
}

bool
ranger_msgs__msg__ActuatorState__are_equal(const ranger_msgs__msg__ActuatorState * lhs, const ranger_msgs__msg__ActuatorState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // motor
  if (!ranger_msgs__msg__MotorState__are_equal(
      &(lhs->motor), &(rhs->motor)))
  {
    return false;
  }
  // driver
  if (!ranger_msgs__msg__DriverState__are_equal(
      &(lhs->driver), &(rhs->driver)))
  {
    return false;
  }
  return true;
}

bool
ranger_msgs__msg__ActuatorState__copy(
  const ranger_msgs__msg__ActuatorState * input,
  ranger_msgs__msg__ActuatorState * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // motor
  if (!ranger_msgs__msg__MotorState__copy(
      &(input->motor), &(output->motor)))
  {
    return false;
  }
  // driver
  if (!ranger_msgs__msg__DriverState__copy(
      &(input->driver), &(output->driver)))
  {
    return false;
  }
  return true;
}

ranger_msgs__msg__ActuatorState *
ranger_msgs__msg__ActuatorState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorState * msg = (ranger_msgs__msg__ActuatorState *)allocator.allocate(sizeof(ranger_msgs__msg__ActuatorState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ranger_msgs__msg__ActuatorState));
  bool success = ranger_msgs__msg__ActuatorState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ranger_msgs__msg__ActuatorState__destroy(ranger_msgs__msg__ActuatorState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ranger_msgs__msg__ActuatorState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ranger_msgs__msg__ActuatorState__Sequence__init(ranger_msgs__msg__ActuatorState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorState * data = NULL;

  if (size) {
    data = (ranger_msgs__msg__ActuatorState *)allocator.zero_allocate(size, sizeof(ranger_msgs__msg__ActuatorState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ranger_msgs__msg__ActuatorState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ranger_msgs__msg__ActuatorState__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
ranger_msgs__msg__ActuatorState__Sequence__fini(ranger_msgs__msg__ActuatorState__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      ranger_msgs__msg__ActuatorState__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

ranger_msgs__msg__ActuatorState__Sequence *
ranger_msgs__msg__ActuatorState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorState__Sequence * array = (ranger_msgs__msg__ActuatorState__Sequence *)allocator.allocate(sizeof(ranger_msgs__msg__ActuatorState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ranger_msgs__msg__ActuatorState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ranger_msgs__msg__ActuatorState__Sequence__destroy(ranger_msgs__msg__ActuatorState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ranger_msgs__msg__ActuatorState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ranger_msgs__msg__ActuatorState__Sequence__are_equal(const ranger_msgs__msg__ActuatorState__Sequence * lhs, const ranger_msgs__msg__ActuatorState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ranger_msgs__msg__ActuatorState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ranger_msgs__msg__ActuatorState__Sequence__copy(
  const ranger_msgs__msg__ActuatorState__Sequence * input,
  ranger_msgs__msg__ActuatorState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ranger_msgs__msg__ActuatorState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ranger_msgs__msg__ActuatorState * data =
      (ranger_msgs__msg__ActuatorState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ranger_msgs__msg__ActuatorState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ranger_msgs__msg__ActuatorState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ranger_msgs__msg__ActuatorState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
