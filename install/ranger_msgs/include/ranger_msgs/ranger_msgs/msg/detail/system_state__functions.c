// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from ranger_msgs:msg/SystemState.idl
// generated code does not contain a copyright notice
#include "ranger_msgs/msg/detail/system_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
ranger_msgs__msg__SystemState__init(ranger_msgs__msg__SystemState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    ranger_msgs__msg__SystemState__fini(msg);
    return false;
  }
  // vehicle_state
  // control_mode
  // error_code
  // battery_voltage
  // motion_mode
  return true;
}

void
ranger_msgs__msg__SystemState__fini(ranger_msgs__msg__SystemState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // vehicle_state
  // control_mode
  // error_code
  // battery_voltage
  // motion_mode
}

bool
ranger_msgs__msg__SystemState__are_equal(const ranger_msgs__msg__SystemState * lhs, const ranger_msgs__msg__SystemState * rhs)
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
  // vehicle_state
  if (lhs->vehicle_state != rhs->vehicle_state) {
    return false;
  }
  // control_mode
  if (lhs->control_mode != rhs->control_mode) {
    return false;
  }
  // error_code
  if (lhs->error_code != rhs->error_code) {
    return false;
  }
  // battery_voltage
  if (lhs->battery_voltage != rhs->battery_voltage) {
    return false;
  }
  // motion_mode
  if (lhs->motion_mode != rhs->motion_mode) {
    return false;
  }
  return true;
}

bool
ranger_msgs__msg__SystemState__copy(
  const ranger_msgs__msg__SystemState * input,
  ranger_msgs__msg__SystemState * output)
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
  // vehicle_state
  output->vehicle_state = input->vehicle_state;
  // control_mode
  output->control_mode = input->control_mode;
  // error_code
  output->error_code = input->error_code;
  // battery_voltage
  output->battery_voltage = input->battery_voltage;
  // motion_mode
  output->motion_mode = input->motion_mode;
  return true;
}

ranger_msgs__msg__SystemState *
ranger_msgs__msg__SystemState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__SystemState * msg = (ranger_msgs__msg__SystemState *)allocator.allocate(sizeof(ranger_msgs__msg__SystemState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(ranger_msgs__msg__SystemState));
  bool success = ranger_msgs__msg__SystemState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
ranger_msgs__msg__SystemState__destroy(ranger_msgs__msg__SystemState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    ranger_msgs__msg__SystemState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
ranger_msgs__msg__SystemState__Sequence__init(ranger_msgs__msg__SystemState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__SystemState * data = NULL;

  if (size) {
    data = (ranger_msgs__msg__SystemState *)allocator.zero_allocate(size, sizeof(ranger_msgs__msg__SystemState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = ranger_msgs__msg__SystemState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        ranger_msgs__msg__SystemState__fini(&data[i - 1]);
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
ranger_msgs__msg__SystemState__Sequence__fini(ranger_msgs__msg__SystemState__Sequence * array)
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
      ranger_msgs__msg__SystemState__fini(&array->data[i]);
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

ranger_msgs__msg__SystemState__Sequence *
ranger_msgs__msg__SystemState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  ranger_msgs__msg__SystemState__Sequence * array = (ranger_msgs__msg__SystemState__Sequence *)allocator.allocate(sizeof(ranger_msgs__msg__SystemState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = ranger_msgs__msg__SystemState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
ranger_msgs__msg__SystemState__Sequence__destroy(ranger_msgs__msg__SystemState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    ranger_msgs__msg__SystemState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
ranger_msgs__msg__SystemState__Sequence__are_equal(const ranger_msgs__msg__SystemState__Sequence * lhs, const ranger_msgs__msg__SystemState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!ranger_msgs__msg__SystemState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
ranger_msgs__msg__SystemState__Sequence__copy(
  const ranger_msgs__msg__SystemState__Sequence * input,
  ranger_msgs__msg__SystemState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(ranger_msgs__msg__SystemState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    ranger_msgs__msg__SystemState * data =
      (ranger_msgs__msg__SystemState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!ranger_msgs__msg__SystemState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          ranger_msgs__msg__SystemState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!ranger_msgs__msg__SystemState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
