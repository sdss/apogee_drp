# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class Apogee_drpError(Exception):
    """A custom core Apogee_drp exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(Apogee_drpError, self).__init__(message)


class Apogee_drpNotImplemented(Apogee_drpError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(Apogee_drpNotImplemented, self).__init__(message)


class Apogee_drpAPIError(Apogee_drpError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Apogee_drp API'
        else:
            message = 'Http response error from Apogee_drp API. {0}'.format(message)

        super(Apogee_drpAPIError, self).__init__(message)


class Apogee_drpApiAuthError(Apogee_drpAPIError):
    """A custom exception for API authentication errors"""
    pass


class Apogee_drpMissingDependency(Apogee_drpError):
    """A custom exception for missing dependencies."""
    pass


class Apogee_drpWarning(Warning):
    """Base warning for Apogee_drp."""


class Apogee_drpUserWarning(UserWarning, Apogee_drpWarning):
    """The primary warning class."""
    pass


class Apogee_drpSkippedTestWarning(Apogee_drpUserWarning):
    """A warning for when a test is skipped."""
    pass


class Apogee_drpDeprecationWarning(Apogee_drpUserWarning):
    """A warning for deprecated features."""
    pass
