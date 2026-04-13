// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ranger_msgs:msg/ActuatorStateArray.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/actuator_state_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `states`
#include "ranger_msgs/msg/detail/actuator_state__functions.h"

bool
ranger_msgs__msg__ActuatorStateArray__init(ranger_msgs__msg__ActuatorStateArray * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    ranger_msgs__msg__ActuatorStateArray__fini(msg);
    return false;
  }
  // states
  if (!ranger_msgs__msg__ActuatorState__Sequence__init(&msg->states, 0)) {
    ranger_msgs__msg__ActuatorStateArray__fini(msg);
    return false;
  }
  return true;
}

void
ranger_msgs__msg__ActuatorStateArray__fini(ranger_msgs__msg__ActuatorStateArray * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // states
  ranger_msgs__msg__ActuatorState__Sequence__fini(&msg->states);
}

bool
ranger_msgs__msg__ActuatorStateArray__are_equal(const ranger_msgs__msg__ActuatorStateArray * lhs, const ranger_msgs__msg__ActuatorStateArray * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // states
  if (!ranger_msgs__msg__ActuatorState__Sequence__are_equal(
      &(lhs->states), &(rhs->states)))
  {
    return false;
  }
  return true;
}

bool
ranger_msgs__msg__ActuatorStateArray__copy(
  const ranger_msgs__msg__ActuatorStateArray * input,
  ranger_msgs__msg__ActuatorStateArray * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // states
  if (!ranger_msgs__msg__ActuatorState__Sequence__copy(
      &(input->states), &(output->states)))
  {
    return false;
  }
  return true;
}

ranger_msgs__msg__ActuatorStateArray *
ranger_msgs__msg__ActuatorStateArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorStateArray * msg = (ranger_msgs__msg__ActuatorStateArray *)allocator.allocate(sizeof(ranger_msgs__msg__ActuatorStateArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ranger_msgs__msg__ActuatorStateArray));
  bool success = ranger_msgs__msg__ActuatorStateArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ranger_msgs__msg__ActuatorStateArray__destroy(ranger_msgs__msg__ActuatorStateArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ranger_msgs__msg__ActuatorStateArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ranger_msgs__msg__ActuatorStateArray__Sequence__init(ranger_msgs__msg__ActuatorStateArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorStateArray * data = NULL;

  if (size) {
    data = (ranger_msgs__msg__ActuatorStateArray *)allocator.zero_allocate(size, sizeof(ranger_msgs__msg__ActuatorStateArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ranger_msgs__msg__ActuatorStateArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ranger_msgs__msg__ActuatorStateArray__fini(&data[i - 1]);
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
ranger_msgs__msg__ActuatorStateArray__Sequence__fini(ranger_msgs__msg__ActuatorStateArray__Sequence * array)
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
      ranger_msgs__msg__ActuatorStateArray__fini(&array->data[i]);
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

ranger_msgs__msg__ActuatorStateArray__Sequence *
ranger_msgs__msg__ActuatorStateArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__ActuatorStateArray__Sequence * array = (ranger_msgs__msg__ActuatorStateArray__Sequence *)allocator.allocate(sizeof(ranger_msgs__msg__ActuatorStateArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ranger_msgs__msg__ActuatorStateArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ranger_msgs__msg__ActuatorStateArray__Sequence__destroy(ranger_msgs__msg__ActuatorStateArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ranger_msgs__msg__ActuatorStateArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ranger_msgs__msg__ActuatorStateArray__Sequence__are_equal(const ranger_msgs__msg__ActuatorStateArray__Sequence * lhs, const ranger_msgs__msg__ActuatorStateArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ranger_msgs__msg__ActuatorStateArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ranger_msgs__msg__ActuatorStateArray__Sequence__copy(
  const ranger_msgs__msg__ActuatorStateArray__Sequence * input,
  ranger_msgs__msg__ActuatorStateArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ranger_msgs__msg__ActuatorStateArray);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ranger_msgs__msg__ActuatorStateArray * data =
      (ranger_msgs__msg__ActuatorStateArray *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ranger_msgs__msg__ActuatorStateArray__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ranger_msgs__msg__ActuatorStateArray__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ranger_msgs__msg__ActuatorStateArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
