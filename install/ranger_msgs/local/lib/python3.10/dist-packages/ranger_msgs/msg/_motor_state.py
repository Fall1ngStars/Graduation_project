# generated from rosidl_generator_py/resource/_idl.py.em
# with input from ranger_msgs:msg/MotorState.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_MotorState(type):
    """Metaclass of message 'MotorState'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('ranger_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'ranger_msgs.msg.MotorState')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__motor_state
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__motor_state
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__motor_state
            cls._TYPE_SUPPORT = module.type_support_msg__msg__motor_state
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__motor_state

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class MotorState(metaclass=Metaclass_MotorState):
    """Message class 'MotorState'."""

    __slots__ = [
        '_rpm',
        '_current',
        '_pulse_count',
        '_motor_angles',
        '_motor_speeds',
    ]

    _fields_and_field_types = {
        'rpm': 'int32',
        'current': 'float',
        'pulse_count': 'int32',
        'motor_angles': 'float',
        'motor_speeds': 'float',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('int32'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.rpm = kwargs.get('rpm', int())
        self.current = kwargs.get('current', float())
        self.pulse_count = kwargs.get('pulse_count', int())
        self.motor_angles = kwargs.get('motor_angles', float())
        self.motor_speeds = kwargs.get('motor_speeds', float())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.rpm != other.rpm:
            return False
        if self.current != other.current:
            return False
        if self.pulse_count != other.pulse_count:
            return False
        if self.motor_angles != other.motor_angles:
            return False
        if self.motor_speeds != other.motor_speeds:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def rpm(self):
        """Message field 'rpm'."""
        return self._rpm

    @rpm.setter
    def rpm(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'rpm' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'rpm' field must be an integer in [-2147483648, 2147483647]"
        self._rpm = value

    @builtins.property
    def current(self):
        """Message field 'current'."""
        return self._current

    @current.setter
    def current(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'current' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'current' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._current = value

    @builtins.property
    def pulse_count(self):
        """Message field 'pulse_count'."""
        return self._pulse_count

    @pulse_count.setter
    def pulse_count(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'pulse_count' field must be of type 'int'"
            assert value >= -2147483648 and value < 2147483648, \
                "The 'pulse_count' field must be an integer in [-2147483648, 2147483647]"
        self._pulse_count = value

    @builtins.property
    def motor_angles(self):
        """Message field 'motor_angles'."""
        return self._motor_angles

    @motor_angles.setter
    def motor_angles(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'motor_angles' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'motor_angles' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._motor_angles = value

    @builtins.property
    def motor_speeds(self):
        """Message field 'motor_speeds'."""
        return self._motor_speeds

    @motor_speeds.setter
    def motor_speeds(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'motor_speeds' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'motor_speeds' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._motor_speeds = value
