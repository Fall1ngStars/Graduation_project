// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from ranger_msgs:msg/MotorState.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "ranger_msgs/msg/detail/motor_state__struct.h"
#include "ranger_msgs/msg/detail/motor_state__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool ranger_msgs__msg__motor_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[40];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("ranger_msgs.msg._motor_state.MotorState", full_classname_dest, 39) == 0);
  }
  ranger_msgs__msg__MotorState * ros_message = _ros_message;
  {  // rpm
    PyObject * field = PyObject_GetAttrString(_pymsg, "rpm");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->rpm = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // current
    PyObject * field = PyObject_GetAttrString(_pymsg, "current");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->current = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // pulse_count
    PyObject * field = PyObject_GetAttrString(_pymsg, "pulse_count");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->pulse_count = (int32_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // motor_angles
    PyObject * field = PyObject_GetAttrString(_pymsg, "motor_angles");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->motor_angles = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // motor_speeds
    PyObject * field = PyObject_GetAttrString(_pymsg, "motor_speeds");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->motor_speeds = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * ranger_msgs__msg__motor_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MotorState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("ranger_msgs.msg._motor_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MotorState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  ranger_msgs__msg__MotorState * ros_message = (ranger_msgs__msg__MotorState *)raw_ros_message;
  {  // rpm
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->rpm);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rpm", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // current
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->current);
    {
      int rc = PyObject_SetAttrString(_pymessage, "current", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // pulse_count
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->pulse_count);
    {
      int rc = PyObject_SetAttrString(_pymessage, "pulse_count", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // motor_angles
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->motor_angles);
    {
      int rc = PyObject_SetAttrString(_pymessage, "motor_angles", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // motor_speeds
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->motor_speeds);
    {
      int rc = PyObject_SetAttrString(_pymessage, "motor_speeds", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
