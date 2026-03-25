########################################################
# Started Logging At: 2026-03-24 15:30:57
########################################################
import sys as _VSCODE_sys
print(_VSCODE_sys.executable); del _VSCODE_sys
try:
    import ipywidgets as _VSCODE_ipywidgets
    print("e976ee50-99ed-4aba-9b6b-9dcd5634d07d:IPyWidgets:" + _VSCODE_ipywidgets.__version__)
    del _VSCODE_ipywidgets
except:
    pass
def __VSCODE_inject_module():

    def __VSCODE_call_function(function, callback, data=None):
        __VSCODE_send_chat_message__(function, data, callback=callback)

    def __VSCODE_send_chat_message__(function, data, callback):
        requests = {}
        try:
            requests = __VSCODE_send_chat_message__.__requests
        except Exception:
            __VSCODE_send_chat_message__.__requests = requests

        import uuid as __VSCODE_send_chat_message__uuid
        import IPython.display as __VSCODE_send_chat_message__ipython_display

        id = str(__VSCODE_send_chat_message__uuid.uuid4())
        requests[id] = callback
        data_is_none = data is None
        __VSCODE_send_chat_message__ipython_display.display({"application/vnd.vscode.chat_message": data}, metadata={"id":id, "function": function, "dataIsNone": data_is_none}, raw=True)

        del __VSCODE_send_chat_message__ipython_display
        del __VSCODE_send_chat_message__uuid

    def __VSCODE_on_chat_message(id, data):
        requests = {}
        try:
            requests = __VSCODE_send_chat_message__.__requests
        except Exception:
            __VSCODE_send_chat_message__.__requests = requests

        if id in requests:
            requests[id](data)
            del requests[id]
        else:
            raise NotImplementedError(f"Callback not found for message {id}")

    import sys as __VSCODE_send_chat_message__sys
    import IPython as __VSCODE_send_chat_message__IPython
    chat = type(__VSCODE_send_chat_message__IPython)("chat")
    chat.send_message = __VSCODE_send_chat_message__
    chat.call_function = __VSCODE_call_function
    chat.__on_message = __VSCODE_on_chat_message
    __VSCODE_send_chat_message__sys.modules["vscode"] = type(__VSCODE_send_chat_message__IPython)("vscode")
    __VSCODE_send_chat_message__sys.modules["vscode"].chat = chat
    del __VSCODE_send_chat_message__sys
    del __VSCODE_send_chat_message__IPython


__VSCODE_inject_module()
del __VSCODE_inject_module

__vsc_ipynb_file__ = "/blue/adamginsburg/adamginsburg/repos/icemodels/examples/co_profiles_bergner_polar_apolar.ipynb"
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import icemodels
import icemodels.co_profiles as co_profiles
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    profiles[label] = {'tau_total': tau_total, 'tau_components': tau_components}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.25)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25)

plt.tight_layout()
# Optional: show transmission e^(-tau) for direct visual comparison
plt.figure(figsize=(7, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    plt.plot(xarr.value, transmission, lw=2, label=label)

plt.xlabel('Wavelength (µm)')
plt.ylabel('Transmission, exp(-τ)')
plt.title('CO Band Transmission vs Polar/Apolar Mixture')
plt.legend(fontsize=9)
plt.grid(alpha=0.25)
plt.tight_layout()
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.0": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.0.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.0.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapz(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapz(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar CO lab table. Run icemodels.download_all_ocdb() first.'
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.1": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.1.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.1.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapz(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapz(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.2": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.2.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.2.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapz(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapz(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8030682910362105 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8314479919468359 ... -0.037706526359304715
#[Out]#  Polar-dominated (20/80) 0.8829068937375829 ...  -0.10290607393773016
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8030682910362105 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8314479919468359 ... -0.037706526359304715
#[Out]#  Polar-dominated (20/80) 0.8829068937375829 ...  -0.10290607393773016
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in matrix_profiles.items():
    ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Optical Depth, τ')
ax.set_title('Matrix Dependence of CO Profile (Bergner-style)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.3": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.3.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.3.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in matrix_profiles.items():
    ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Optical Depth, τ')
ax.set_title('Matrix Dependence of CO Profile (Bergner-style)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8030682910362105 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8314479919468359 ... -0.037706526359304715
#[Out]#  Polar-dominated (20/80) 0.8829068937375829 ...  -0.10290607393773016
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8030682910362105 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8314479919468359 ... -0.037706526359304715
#[Out]#  Polar-dominated (20/80) 0.8829068937375829 ...  -0.10290607393773016
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in matrix_profiles.items():
    ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Optical Depth, τ')
ax.set_title('Matrix Dependence of CO Profile (Bergner-style)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09165777563000484
#[Out]# H2O matrix (polar CO) 0.9839700108113411 ...                     0.0
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps

co_profiles = importlib.reload(co_profiles)
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.4": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.4.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.4.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

comparison_profiles = {'ocdb': {}, 'bergner': {}}
comparison_rows = []

for db in ('ocdb', 'bergner'):
    for label, cols in matrix_cases.items():
        tau_total, tau_components = co_profiles.co_composite_tau(
            cols,
            xarr,
            database=db,
            temperature=10,
            use_gaussian_fallback=False,
        )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[db][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': db,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for ax, db in zip(axes, ('ocdb', 'bergner')):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{db.upper()} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

comparison_profiles = {'ocdb': {}, 'bergner': {}}
comparison_rows = []

for db in ('ocdb', 'bergner'):
    for label, cols in matrix_cases.items():
        tau_total, tau_components = co_profiles.co_composite_tau(
            cols,
            xarr,
            database=db,
            temperature=10,
            use_gaussian_fallback=False,
        )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[db][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': db,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=4>
#[Out]# database      matrix_case        F466N_rel_flux      F466N_delta_mag   
#[Out]#                                                                        
#[Out]#   str7           str21              float64              float64       
#[Out]# -------- --------------------- ------------------ ---------------------
#[Out]#  bergner   CO2 matrix (CO:CO2) 0.9999929097968144 7.698117588302532e-06
#[Out]#  bergner H2O matrix (polar CO) 0.9986772199426089 0.0014371409196070413
#[Out]#     ocdb   CO2 matrix (CO:CO2)  0.904312954094962   0.10920311984032616
#[Out]#     ocdb H2O matrix (polar CO) 0.9844636769100753  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for ax, db in zip(axes, ('ocdb', 'bergner')):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{db.upper()} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=2>
#[Out]# database H2O_F466N_rel_flux ... CO2_over_H2O_flux  Delta_mag_CO2_minus_H2O
#[Out]#                             ...                                           
#[Out]#   str7        float64       ...      float64               float64        
#[Out]# -------- ------------------ ... ------------------ -----------------------
#[Out]#  bergner 0.9986772199426089 ... 1.0013174325276801  -0.0014294428020187387
#[Out]#     ocdb 0.9844636769100753 ... 0.9185843777734072     0.09220236209084234
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps

co_profiles = importlib.reload(co_profiles)
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

comparison_profiles = {'ocdb': {}, 'bergner': {}}
comparison_rows = []

for db in ('ocdb', 'bergner'):
    for label, cols in matrix_cases.items():
        tau_total, tau_components = co_profiles.co_composite_tau(
            cols,
            xarr,
            database=db,
            temperature=10,
            use_gaussian_fallback=False,
        )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[db][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': db,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=4>
#[Out]# database      matrix_case        F466N_rel_flux      F466N_delta_mag   
#[Out]#                                                                        
#[Out]#   str7           str21              float64              float64       
#[Out]# -------- --------------------- ------------------ ---------------------
#[Out]#  bergner   CO2 matrix (CO:CO2) 0.9999929097968144 7.698117588302532e-06
#[Out]#  bergner H2O matrix (polar CO) 0.9986772199426089 0.0014371409196070413
#[Out]#     ocdb   CO2 matrix (CO:CO2)  0.904312954094962   0.10920311984032616
#[Out]#     ocdb H2O matrix (polar CO) 0.9844636769100753  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for ax, db in zip(axes, ('ocdb', 'bergner')):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{db.upper()} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=2>
#[Out]# database H2O_F466N_rel_flux ... CO2_over_H2O_flux  Delta_mag_CO2_minus_H2O
#[Out]#                             ...                                           
#[Out]#   str7        float64       ...      float64               float64        
#[Out]# -------- ------------------ ... ------------------ -----------------------
#[Out]#  bergner 0.9986772199426089 ... 1.0013174325276801  -0.0014294428020187387
#[Out]#     ocdb 0.9844636769100753 ... 0.9185843777734072     0.09220236209084234
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.5": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.5.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.5.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        tau_total, tau_components = co_profiles.co_composite_tau(
            cols,
            xarr,
            database=db,
            temperature=10,
            use_gaussian_fallback=False,
        )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='bergner',
        temperature=10,
        use_gaussian_fallback=False,
    )
    peak_ocdb = np.nanmax(tau_ocdb)
    peak_bergner = np.nanmax(tau_bergner)
    component_scale[env] = peak_ocdb / peak_bergner

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_scaled += component_scale[env] * tau_comp
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag   
#[Out]#                                           ...                      
#[Out]#        str19                str21         ...        float64       
#[Out]# ------------------- --------------------- ... ---------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ...   0.06327311949540282
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...  0.017604525776557953
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ... 7.698117588302532e-06
#[Out]#         bergner_raw H2O matrix (polar CO) ... 0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...   0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched  0.983916377802644 ...    0.045668593718844866
#[Out]#         bergner_raw 0.9986772199426089 ...  -0.0014294428020187387
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

def smooth_profile(y, window=31):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='same')

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        tau_total, tau_components = co_profiles.co_composite_tau(
            cols,
            xarr,
            database=db,
            temperature=10,
            use_gaussian_fallback=False,
        )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='bergner',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner_sm = smooth_profile(tau_bergner, window=31)
    amp_ocdb = np.nanpercentile(tau_ocdb, 99.5)
    amp_bergner = np.nanpercentile(tau_bergner_sm, 99.5)
    raw_scale = amp_ocdb / max(amp_bergner, 1e-8)
    component_scale[env] = min(raw_scale, 25.0)

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner robust peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles using smoothed components
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_comp_sm = smooth_profile(tau_comp, window=31)
        tau_scaled += component_scale[env] * tau_comp_sm
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag    
#[Out]#                                           ...                       
#[Out]#        str19                str21         ...        float64        
#[Out]# ------------------- --------------------- ... ----------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ... 0.00019242539538166954
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...   0.017695559728828438
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ...  7.698117588302532e-06
#[Out]#         bergner_raw H2O matrix (polar CO) ...  0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...    0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...   0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (robust peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched 0.9838338844296854 ...    -0.01750313433344677
#[Out]#         bergner_raw 0.9986772199426089 ...  -0.0014294428020187387
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

def smooth_profile(y, window=31):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='same')

# Explicit Bergner files for matrix comparison (avoid weak/noisy auto picks)
bergner_cache = icemodels.core.optical_constants_cache_dir
bergner_file_map = {
    'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
    'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
    'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
}
for env, fn in bergner_file_map.items():
    assert os.path.exists(fn), f'Missing Bergner file for {env}: {fn}'
bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}
print('Using explicit Bergner files:')
for env, fn in bergner_file_map.items():
    print(f'  {env:>5}: {os.path.basename(fn)}')

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        if db == 'bergner':
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                tables=bergner_tables,
                use_gaussian_fallback=False,
            )
        else:
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                database='ocdb',
                temperature=10,
                use_gaussian_fallback=False,
            )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        tables={env: bergner_tables[env]},
        use_gaussian_fallback=False,
    )
    tau_bergner_sm = smooth_profile(tau_bergner, window=31)
    amp_ocdb = np.nanpercentile(tau_ocdb, 99.5)
    amp_bergner = np.nanpercentile(tau_bergner_sm, 99.5)
    raw_scale = amp_ocdb / max(amp_bergner, 1e-8)
    component_scale[env] = min(raw_scale, 25.0)

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner robust peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles using smoothed components
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_comp_sm = smooth_profile(tau_comp, window=31)
        tau_scaled += component_scale[env] * tau_comp_sm
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag   
#[Out]#                                           ...                      
#[Out]#        str19                str21         ...        float64       
#[Out]# ------------------- --------------------- ... ---------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ...  0.034834689598815916
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...  0.017695559728828438
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ...   0.00538756711919639
#[Out]#         bergner_raw H2O matrix (polar CO) ... 0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...   0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (robust peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched 0.9838338844296854 ...    0.017139129869987477
#[Out]#         bergner_raw 0.9986772199426089 ...   0.0039504261995893485
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps

co_profiles = importlib.reload(co_profiles)
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

def smooth_profile(y, window=31):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='same')

# Explicit Bergner files for matrix comparison (avoid weak/noisy auto picks)
bergner_cache = icemodels.core.optical_constants_cache_dir
bergner_file_map = {
    'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
    'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
    'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
}
for env, fn in bergner_file_map.items():
    assert os.path.exists(fn), f'Missing Bergner file for {env}: {fn}'
bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}
print('Using explicit Bergner files:')
for env, fn in bergner_file_map.items():
    print(f'  {env:>5}: {os.path.basename(fn)}')

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        if db == 'bergner':
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                tables=bergner_tables,
                use_gaussian_fallback=False,
            )
        else:
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                database='ocdb',
                temperature=10,
                use_gaussian_fallback=False,
            )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        tables={env: bergner_tables[env]},
        use_gaussian_fallback=False,
    )
    tau_bergner_sm = smooth_profile(tau_bergner, window=31)
    amp_ocdb = np.nanpercentile(tau_ocdb, 99.5)
    amp_bergner = np.nanpercentile(tau_bergner_sm, 99.5)
    raw_scale = amp_ocdb / max(amp_bergner, 1e-8)
    component_scale[env] = min(raw_scale, 25.0)

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner robust peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles using smoothed components
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_comp_sm = smooth_profile(tau_comp, window=31)
        tau_scaled += component_scale[env] * tau_comp_sm
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag   
#[Out]#                                           ...                      
#[Out]#        str19                str21         ...        float64       
#[Out]# ------------------- --------------------- ... ---------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ...  0.034834689598815916
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...  0.017695559728828438
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ...   0.00538756711919639
#[Out]#         bergner_raw H2O matrix (polar CO) ... 0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...   0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (robust peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched 0.9838338844296854 ...    0.017139129869987477
#[Out]#         bergner_raw 0.9986772199426089 ...   0.0039504261995893485
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps

co_profiles = importlib.reload(co_profiles)
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

def smooth_profile(y, window=31):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='same')

# Explicit Bergner files for matrix comparison (avoid weak/noisy auto picks)
bergner_cache = icemodels.core.optical_constants_cache_dir
bergner_file_map = {
    'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
    'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
    'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
}
for env, fn in bergner_file_map.items():
    assert os.path.exists(fn), f'Missing Bergner file for {env}: {fn}'
bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}
print('Using explicit Bergner files:')
for env, fn in bergner_file_map.items():
    print(f'  {env:>5}: {os.path.basename(fn)}')

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        if db == 'bergner':
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                tables=bergner_tables,
                use_gaussian_fallback=False,
            )
        else:
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                database='ocdb',
                temperature=10,
                use_gaussian_fallback=False,
            )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        tables={env: bergner_tables[env]},
        use_gaussian_fallback=False,
    )
    tau_bergner_sm = smooth_profile(tau_bergner, window=31)
    amp_ocdb = np.nanpercentile(tau_ocdb, 99.5)
    amp_bergner = np.nanpercentile(tau_bergner_sm, 99.5)
    raw_scale = amp_ocdb / max(amp_bergner, 1e-8)
    component_scale[env] = min(raw_scale, 25.0)

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner robust peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles using smoothed components
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_comp_sm = smooth_profile(tau_comp, window=31)
        tau_scaled += component_scale[env] * tau_comp_sm
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag   
#[Out]#                                           ...                      
#[Out]#        str19                str21         ...        float64       
#[Out]# ------------------- --------------------- ... ---------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ...  0.034834689598815916
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...  0.017695559728828438
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ...   0.00538756711919639
#[Out]#         bergner_raw H2O matrix (polar CO) ... 0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...   0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (robust peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched 0.9838338844296854 ...    0.017139129869987477
#[Out]#         bergner_raw 0.9986772199426089 ...   0.0039504261995893485
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.6": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.6.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.6.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
# Build/refresh Bergner explicit tables if needed
if 'bergner_tables' not in globals():
    _ = co_profiles.download_bergner_co_profiles(redo=False)
    bergner_cache = icemodels.core.optical_constants_cache_dir
    bergner_file_map = {
        'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
        'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
        'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
    }
    bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}

def co_fraction_from_ocdb_table(tbl):
    mols, ratios = icemodels.core.parse_molscomps(tbl.meta['composition'])
    total = np.sum(ratios)
    for mol, ratio in zip(mols, ratios):
        if mol.strip().upper() == 'CO':
            return float(ratio) / float(total)
    raise ValueError(f"No CO component found in composition: {tbl.meta['composition']}")

# CO fractions used in fixed-total-ice mode
co_fraction = {
    'ocdb': {
        'H2O matrix (polar CO)': co_fraction_from_ocdb_table(tables['polar']),
        'CO2 matrix (CO:CO2)': co_fraction_from_ocdb_table(tables['CO2']),
    },
    # For explicit Bergner files: Polar-10-1-1 and Apolar-1-10
    # Interpreted as component ratios with CO fraction 1/(10+1+1) and 1/(1+10).
    'bergner': {
        'H2O matrix (polar CO)': 1.0 / 12.0,
        'CO2 matrix (CO:CO2)': 1.0 / 11.0,
    },
}

N_CO_fixed = 4.0e17 * u.cm**-2
N_ice_fixed = 1.0e19 * u.cm**-2

norm_modes = ('fixed_co', 'fixed_total_ice')
norm_profiles = {mode: {'ocdb': {}, 'bergner': {}} for mode in norm_modes}
norm_rows = []

for mode in norm_modes:
    for db in ('ocdb', 'bergner'):
        for case in ('H2O matrix (polar CO)', 'CO2 matrix (CO:CO2)'):
            if mode == 'fixed_co':
                N_co_case = N_CO_fixed
            else:
                N_co_case = co_fraction[db][case] * N_ice_fixed

            cols = {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': 0.0 * u.cm**-2}
            if case == 'H2O matrix (polar CO)':
                cols['polar'] = N_co_case
            else:
                cols['CO2'] = N_co_case

            if db == 'ocdb':
                tau_total, _ = co_profiles.co_composite_tau(
                    cols, xarr, database='ocdb', temperature=10, use_gaussian_fallback=False
                )
            else:
                tau_total, _ = co_profiles.co_composite_tau(
                    cols, xarr, tables=bergner_tables, use_gaussian_fallback=False
                )

            f_rel = f466n_relative_flux_from_tau(tau_total)
            dmag = -2.5 * np.log10(f_rel)
            norm_profiles[mode][db][case] = {'tau_total': tau_total, 'f466n_rel_flux': f_rel, 'dmag': dmag, 'N_co': N_co_case}

            norm_rows.append({
                'mode': mode,
                'database': db,
                'matrix_case': case,
                'f_CO_for_fixed_total_ice': co_fraction[db][case],
                'N_CO_used_cm-2': N_co_case.to_value(u.cm**-2),
                'tau_peak': float(np.nanmax(tau_total)),
                'F466N_rel_flux': f_rel,
                'F466N_delta_mag': dmag,
            })

norm_table = Table(rows=norm_rows)
norm_table.sort(['mode', 'database', 'matrix_case'])
norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.7": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.7.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.7.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
layout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]

for ax, (mode, db) in zip(axes.flat, layout):
    for case, data in norm_profiles[mode][db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)

    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'
    ax.set_title(f'{db.upper()} | {title_mode}')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)
    axb.set_ylim(0, 1.1)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')

plt.tight_layout()

norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.8": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.8.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.8.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table

import icemodels
import icemodels.co_profiles as co_profiles
from astroquery.svo_fps import SvoFps

co_profiles = importlib.reload(co_profiles)
# Wavelength window around the 4.67 µm CO stretching mode
xarr = np.linspace(4.60, 4.75, 4000) * u.um

# F466N transmission curve for overlays and in-band flux calculation
f466n_id = 'JWST/NIRCam.F466N'
f466n_trans = SvoFps.get_transmission_data(f466n_id)
f466n_wave_um = u.Quantity(f466n_trans['Wavelength']).to(u.um)
f466n_thr = np.array(f466n_trans['Transmission'])
f466n_thr_norm = f466n_thr / np.nanmax(f466n_thr)

# Interpolate filter throughput onto model grid
f466n_thr_on_grid = np.interp(
    xarr.to_value(u.um),
    f466n_wave_um.to_value(u.um),
    f466n_thr,
    left=0.0,
    right=0.0,
    )

def f466n_relative_flux_from_tau(tau):
    transmission = np.exp(-tau)
    num = np.trapezoid(transmission * f466n_thr_on_grid, xarr.to_value(u.um))
    den = np.trapezoid(f466n_thr_on_grid, xarr.to_value(u.um))
    return num / den

# Show the actual files selected for each component at target T=10K
for env in ['pure', 'polar', 'CO2']:
    candidates = co_profiles.find_co_mixture_files(env, database='ocdb', temperature=10)
    chosen = os.path.basename(candidates[0]) if candidates else 'NONE'
    print(f"{env:>5} selected file: {chosen}")

# Load lab optical constants for each environment from OCDB
# (no Gaussian fallback in this notebook)
tables = {
    'pure': co_profiles.load_co_environment('pure', database='ocdb', temperature=10),
    'polar': co_profiles.load_co_environment('polar', database='ocdb', temperature=10),
    'CO2': co_profiles.load_co_environment('CO2', database='ocdb', temperature=10),
}

available = {k: (v is not None) for k, v in tables.items()}
print('Loaded lab tables:', available)

assert tables['pure'] is not None, 'Missing pure/apolar CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['polar'] is not None, 'Missing polar (H2O-matrix) CO lab table. Run icemodels.download_all_ocdb() first.'
assert tables['CO2'] is not None, 'Missing CO2-matrix CO lab table. Run icemodels.download_all_ocdb() first.'
# Keep total CO column fixed while shifting partition between apolar and polar environments
N_total = 1.0e18 * u.cm**-2
N_co2 = 0.8e17 * u.cm**-2

mixtures = {
    'Apolar-dominated (80/20)': {'pure': 0.80 * N_total, 'polar': 0.20 * N_total, 'CO2': N_co2},
    'Intermediate (50/50)': {'pure': 0.50 * N_total, 'polar': 0.50 * N_total, 'CO2': N_co2},
    'Polar-dominated (20/80)': {'pure': 0.20 * N_total, 'polar': 0.80 * N_total, 'CO2': N_co2},
}

profiles = {}
for label, cols in mixtures.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

# Relative comparison table (vs apolar-dominated baseline)
baseline_label = 'Apolar-dominated (80/20)'
baseline_flux = profiles[baseline_label]['f466n_rel_flux']

mix_rows = []
for label, data in profiles.items():
    rel_to_baseline = data['f466n_rel_flux'] / baseline_flux
    mix_rows.append({
        'mixture': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_baseline': rel_to_baseline,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_baseline': data['f466n_delta_mag'] - profiles[baseline_label]['f466n_delta_mag'],
    })

mix_table = Table(rows=mix_rows)
mix_table.sort('mixture')
mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: total profile for each mixture
for label, data in profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_title('Total CO Optical Depth')
axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: component breakdown for one representative mixture
rep_label = 'Intermediate (50/50)'
rep = profiles[rep_label]
for env_name, tau_comp in rep['tau_components'].items():
    axes[1].plot(xarr.value, tau_comp, lw=2, label=env_name)
axes[1].plot(xarr.value, rep['tau_total'], 'k--', lw=2, label='total')

axes[1].set_title(f'Component Decomposition: {rep_label}')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25)

# Overlay F466N transmission on secondary y-axis
ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()
# Show transmission e^(-tau) for each mixture, with F466N bandpass overlay
fig, ax = plt.subplots(figsize=(8, 5))
for label, data in profiles.items():
    transmission = np.exp(-data['tau_total'])
    ax.plot(xarr.value, transmission, lw=2, label=label)

ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Transmission, exp(-τ)')
ax.set_title('CO Band Transmission vs Polar/Apolar Mixture')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25)

axb = ax.twinx()
axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
axb.set_ylabel('F466N transmission (norm.)', color='gray')
axb.tick_params(axis='y', colors='gray')
axb.set_ylim(0, 1.1)

plt.tight_layout()

mix_table
#[Out]# <Table length=3>
#[Out]#         mixture            F466N_rel_flux   ... Delta_mag_vs_baseline
#[Out]#                                             ...                      
#[Out]#          str24                float64       ...        float64       
#[Out]# ------------------------ ------------------ ... ---------------------
#[Out]# Apolar-dominated (80/20) 0.8040902864467462 ...                   0.0
#[Out]#     Intermediate (50/50) 0.8336155210017514 ...  -0.03915244080413938
#[Out]#  Polar-dominated (20/80) 0.8853024220389041 ...  -0.10446709146168448
N_matrix = 4.0e17 * u.cm**-2

matrix_cases = {
    'H2O matrix (polar CO)': {'pure': 0.0 * u.cm**-2, 'polar': N_matrix, 'CO2': 0.0 * u.cm**-2},
    'CO2 matrix (CO:CO2)': {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': N_matrix},
}

matrix_profiles = {}
for label, cols in matrix_cases.items():
    tau_total, tau_components = co_profiles.co_composite_tau(
        cols,
        xarr,
        tables=tables,
        use_gaussian_fallback=False,
    )
    f_rel = f466n_relative_flux_from_tau(tau_total)
    delta_mag = -2.5 * np.log10(f_rel)
    matrix_profiles[label] = {
        'tau_total': tau_total,
        'tau_components': tau_components,
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }

matrix_rows = []
base_label = 'H2O matrix (polar CO)'
base_flux = matrix_profiles[base_label]['f466n_rel_flux']
base_mag = matrix_profiles[base_label]['f466n_delta_mag']

for label, data in matrix_profiles.items():
    matrix_rows.append({
        'matrix_case': label,
        'F466N_rel_flux': data['f466n_rel_flux'],
        'F466N_rel_to_H2O_matrix': data['f466n_rel_flux'] / base_flux,
        'F466N_delta_mag': data['f466n_delta_mag'],
        'Delta_mag_vs_H2O_matrix': data['f466n_delta_mag'] - base_mag,
    })

matrix_table = Table(rows=matrix_rows)
matrix_table.sort('matrix_case')
matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# Left: linear scale (absolute optical depth)
for label, data in matrix_profiles.items():
    axes[0].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[0].set_xlabel('Wavelength (µm)')
axes[0].set_ylabel('Optical Depth, τ')
axes[0].set_title('Matrix Dependence of CO Profile (linear)')
axes[0].legend(fontsize=9, loc='upper left')
axes[0].grid(alpha=0.25)

ax0b = axes[0].twinx()
ax0b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax0b.set_ylabel('F466N transmission (norm.)', color='gray')
ax0b.tick_params(axis='y', colors='gray')
ax0b.set_ylim(0, 1.1)

# Right: log scale to show weaker H2O:CO peak clearly
for label, data in matrix_profiles.items():
    axes[1].plot(xarr.value, data['tau_total'], lw=2, label=label)

axes[1].set_yscale('log')
axes[1].set_xlabel('Wavelength (µm)')
axes[1].set_ylabel('Optical Depth, τ (log scale)')
axes[1].set_title('Matrix Dependence of CO Profile (log)')
axes[1].legend(fontsize=9, loc='upper left')
axes[1].grid(alpha=0.25, which='both')

ax1b = axes[1].twinx()
ax1b.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
ax1b.set_ylabel('F466N transmission (norm.)', color='gray')
ax1b.tick_params(axis='y', colors='gray')
ax1b.set_ylim(0, 1.1)

plt.tight_layout()

matrix_table
#[Out]# <Table length=2>
#[Out]#      matrix_case        F466N_rel_flux   ... Delta_mag_vs_H2O_matrix
#[Out]#                                          ...                        
#[Out]#         str21              float64       ...         float64        
#[Out]# --------------------- ------------------ ... -----------------------
#[Out]#   CO2 matrix (CO:CO2)  0.904312954094962 ...     0.09220236209084234
#[Out]# H2O matrix (polar CO) 0.9844636769100753 ...                     0.0
import importlib
co_profiles = importlib.reload(co_profiles)
assert hasattr(co_profiles, 'download_bergner_co_profiles'), 'co_profiles is outdated in kernel; reload failed'

# Ensure Bergner profiles are available locally (no-op if already downloaded)
_ = co_profiles.download_bergner_co_profiles(redo=False)

def smooth_profile(y, window=31):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(y, kernel, mode='same')

# Explicit Bergner files for matrix comparison (avoid weak/noisy auto picks)
bergner_cache = icemodels.core.optical_constants_cache_dir
bergner_file_map = {
    'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
    'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
    'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
}
for env, fn in bergner_file_map.items():
    assert os.path.exists(fn), f'Missing Bergner file for {env}: {fn}'
bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}
print('Using explicit Bergner files:')
for env, fn in bergner_file_map.items():
    print(f'  {env:>5}: {os.path.basename(fn)}')

comparison_profiles = {'ocdb': {}, 'bergner_raw': {}, 'bergner_peakmatched': {}}
comparison_rows = []

# Build OCDB and raw Bergner profiles for the same matrix cases
for db in ('ocdb', 'bergner'):
    store_key = db if db == 'ocdb' else 'bergner_raw'
    for label, cols in matrix_cases.items():
        if db == 'bergner':
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                tables=bergner_tables,
                use_gaussian_fallback=False,
            )
        else:
            tau_total, tau_components = co_profiles.co_composite_tau(
                cols,
                xarr,
                database='ocdb',
                temperature=10,
                use_gaussian_fallback=False,
            )
        f_rel = f466n_relative_flux_from_tau(tau_total)
        delta_mag = -2.5 * np.log10(f_rel)
        comparison_profiles[store_key][label] = {
            'tau_total': tau_total,
            'tau_components': tau_components,
            'f466n_rel_flux': f_rel,
            'f466n_delta_mag': delta_mag,
        }
        comparison_rows.append({
            'database': store_key,
            'matrix_case': label,
            'F466N_rel_flux': f_rel,
            'F466N_delta_mag': delta_mag,
        })

# Derive component-wise scale factors so Bergner and OCDB have comparable amplitude
N_ref = 4.0e17 * u.cm**-2
component_scale = {}
for env in ('pure', 'polar', 'CO2'):
    tau_ocdb, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        database='ocdb',
        temperature=10,
        use_gaussian_fallback=False,
    )
    tau_bergner, _ = co_profiles.co_composite_tau(
        {env: N_ref},
        xarr,
        tables={env: bergner_tables[env]},
        use_gaussian_fallback=False,
    )
    tau_bergner_sm = smooth_profile(tau_bergner, window=31)
    amp_ocdb = np.nanpercentile(tau_ocdb, 99.5)
    amp_bergner = np.nanpercentile(tau_bergner_sm, 99.5)
    raw_scale = amp_ocdb / max(amp_bergner, 1e-8)
    component_scale[env] = min(raw_scale, 25.0)

scale_table = Table(rows=[
    {'component': k, 'peak_scale_ocdb_over_bergner': v}
    for k, v in component_scale.items()
], names=['component', 'peak_scale_ocdb_over_bergner'])
scale_table.sort('component')
print('Applied Bergner robust peak-matching scale factors (to OCDB):')
scale_table

# Build peak-matched Bergner matrix-case profiles using smoothed components
for label, data in comparison_profiles['bergner_raw'].items():
    tau_scaled = np.zeros_like(data['tau_total'])
    for env, tau_comp in data['tau_components'].items():
        tau_comp_sm = smooth_profile(tau_comp, window=31)
        tau_scaled += component_scale[env] * tau_comp_sm
    f_rel = f466n_relative_flux_from_tau(tau_scaled)
    delta_mag = -2.5 * np.log10(f_rel)
    comparison_profiles['bergner_peakmatched'][label] = {
        'tau_total': tau_scaled,
        'tau_components': data['tau_components'],
        'f466n_rel_flux': f_rel,
        'f466n_delta_mag': delta_mag,
    }
    comparison_rows.append({
        'database': 'bergner_peakmatched',
        'matrix_case': label,
        'F466N_rel_flux': f_rel,
        'F466N_delta_mag': delta_mag,
    })

comparison_table = Table(rows=comparison_rows)
comparison_table.sort(['database', 'matrix_case'])
comparison_table
#[Out]# <Table length=6>
#[Out]#       database           matrix_case      ...    F466N_delta_mag   
#[Out]#                                           ...                      
#[Out]#        str19                str21         ...        float64       
#[Out]# ------------------- --------------------- ... ---------------------
#[Out]# bergner_peakmatched   CO2 matrix (CO:CO2) ...  0.034834689598815916
#[Out]# bergner_peakmatched H2O matrix (polar CO) ...  0.017695559728828438
#[Out]#         bergner_raw   CO2 matrix (CO:CO2) ...   0.00538756711919639
#[Out]#         bergner_raw H2O matrix (polar CO) ... 0.0014371409196070413
#[Out]#                ocdb   CO2 matrix (CO:CO2) ...   0.10920311984032616
#[Out]#                ocdb H2O matrix (polar CO) ...  0.017000757749483817
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

panel_map = [('ocdb', 'OCDB'), ('bergner_peakmatched', 'BERGNER (robust peak-matched to OCDB)')]
for ax, (db, title) in zip(axes, panel_map):
    for label, data in comparison_profiles[db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=label)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.set_title(f'{title} matrix comparison')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.5, alpha=0.9)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')
    axb.set_ylim(0, 1.1)

plt.tight_layout()

# Add per-database relative summary (CO2 matrix relative to H2O matrix)
summary_rows = []
for db in ('ocdb', 'bergner_raw', 'bergner_peakmatched'):
    h2o = comparison_profiles[db]['H2O matrix (polar CO)']
    co2 = comparison_profiles[db]['CO2 matrix (CO:CO2)']
    summary_rows.append({
        'database': db,
        'H2O_F466N_rel_flux': h2o['f466n_rel_flux'],
        'CO2_F466N_rel_flux': co2['f466n_rel_flux'],
        'CO2_over_H2O_flux': co2['f466n_rel_flux'] / h2o['f466n_rel_flux'],
        'Delta_mag_CO2_minus_H2O': co2['f466n_delta_mag'] - h2o['f466n_delta_mag'],
    })

comparison_summary = Table(rows=summary_rows)
comparison_summary.sort('database')
comparison_summary
#[Out]# <Table length=3>
#[Out]#       database      H2O_F466N_rel_flux ... Delta_mag_CO2_minus_H2O
#[Out]#                                        ...                        
#[Out]#        str19             float64       ...         float64        
#[Out]# ------------------- ------------------ ... -----------------------
#[Out]# bergner_peakmatched 0.9838338844296854 ...    0.017139129869987477
#[Out]#         bergner_raw 0.9986772199426089 ...   0.0039504261995893485
#[Out]#                ocdb 0.9844636769100753 ...     0.09220236209084234
# Build/refresh Bergner explicit tables if needed
if 'bergner_tables' not in globals():
    _ = co_profiles.download_bergner_co_profiles(redo=False)
    bergner_cache = icemodels.core.optical_constants_cache_dir
    bergner_file_map = {
        'pure': os.path.join(bergner_cache, 'bergner_13948083_CO_10K.txt'),
        'polar': os.path.join(bergner_cache, 'bergner_13948069_Polar-10-1-1_10K.txt'),
        'CO2': os.path.join(bergner_cache, 'bergner_13948083_Apolar-1-10_10K.txt'),
    }
    bergner_tables = {env: co_profiles.read_bergner_file(fn) for env, fn in bergner_file_map.items()}

def co_fraction_from_ocdb_table(tbl):
    mols, ratios = icemodels.core.parse_molscomps(tbl.meta['composition'])
    total = np.sum(ratios)
    for mol, ratio in zip(mols, ratios):
        if mol.strip().upper() == 'CO':
            return float(ratio) / float(total)
    raise ValueError(f"No CO component found in composition: {tbl.meta['composition']}")

# CO fractions used in fixed-total-ice mode
co_fraction = {
    'ocdb': {
        'H2O matrix (polar CO)': co_fraction_from_ocdb_table(tables['polar']),
        'CO2 matrix (CO:CO2)': co_fraction_from_ocdb_table(tables['CO2']),
    },
    # For explicit Bergner files: Polar-10-1-1 and Apolar-1-10
    # Interpreted as component ratios with CO fraction 1/(10+1+1) and 1/(1+10).
    'bergner': {
        'H2O matrix (polar CO)': 1.0 / 12.0,
        'CO2 matrix (CO:CO2)': 1.0 / 11.0,
    },
}

N_CO_fixed = 4.0e17 * u.cm**-2
N_ice_fixed = 1.0e19 * u.cm**-2

norm_modes = ('fixed_co', 'fixed_total_ice')
norm_profiles = {mode: {'ocdb': {}, 'bergner': {}} for mode in norm_modes}
norm_rows = []

for mode in norm_modes:
    for db in ('ocdb', 'bergner'):
        for case in ('H2O matrix (polar CO)', 'CO2 matrix (CO:CO2)'):
            if mode == 'fixed_co':
                N_co_case = N_CO_fixed
            else:
                N_co_case = co_fraction[db][case] * N_ice_fixed

            cols = {'pure': 0.0 * u.cm**-2, 'polar': 0.0 * u.cm**-2, 'CO2': 0.0 * u.cm**-2}
            if case == 'H2O matrix (polar CO)':
                cols['polar'] = N_co_case
            else:
                cols['CO2'] = N_co_case

            if db == 'ocdb':
                tau_total, _ = co_profiles.co_composite_tau(
                    cols, xarr, database='ocdb', temperature=10, use_gaussian_fallback=False
                )
            else:
                tau_total, _ = co_profiles.co_composite_tau(
                    cols, xarr, tables=bergner_tables, use_gaussian_fallback=False
                )

            f_rel = f466n_relative_flux_from_tau(tau_total)
            dmag = -2.5 * np.log10(f_rel)
            norm_profiles[mode][db][case] = {'tau_total': tau_total, 'f466n_rel_flux': f_rel, 'dmag': dmag, 'N_co': N_co_case}

            norm_rows.append({
                'mode': mode,
                'database': db,
                'matrix_case': case,
                'f_CO_for_fixed_total_ice': co_fraction[db][case],
                'N_CO_used_cm-2': N_co_case.to_value(u.cm**-2),
                'tau_peak': float(np.nanmax(tau_total)),
                'F466N_rel_flux': f_rel,
                'F466N_delta_mag': dmag,
            })

norm_table = Table(rows=norm_rows)
norm_table.sort(['mode', 'database', 'matrix_case'])
norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
layout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]

for ax, (mode, db) in zip(axes.flat, layout):
    for case, data in norm_profiles[mode][db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)

    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'
    ax.set_title(f'{db.upper()} | {title_mode}')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)
    axb.set_ylim(0, 1.1)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')

plt.tight_layout()

norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
layout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]

for ax, (mode, db) in zip(axes.flat, layout):
    for case, data in norm_profiles[mode][db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)

    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'
    ax.set_title(f'{db.upper()} | {title_mode}')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)
    axb.set_ylim(0, 0.1)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')

plt.tight_layout()

norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.9": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 631)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.9.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.9.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.10": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 633)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.10.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.10.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.11": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set_clip_box", 19, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.11.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.11.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.12": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set", 10, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.12.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.12.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.13": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set_clip_box", 19, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.13.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.13.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.14": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set_ybound", 17, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.14.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.14.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.15": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set_ylabel", 17, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.15.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.15.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.16": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_inspect("fig, a.set_ylim", 15, 0)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.16.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.16.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.17": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.set_ylim(0, 0.\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 647)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.17.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.17.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.18": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.set_ylim(0, 0.1\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 647)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.18.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.18.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.19": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.set_ylim(0, 0.1\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 647)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.19.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.19.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.20": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.set_ylim(0, 0.1\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 647)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.20.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.20.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.21": ""}, raw=True, display_id=True)

    def do_implementation():
        return get_ipython().kernel.do_complete("fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)\nlayout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]\n\nfor ax, (mode, db) in zip(axes.flat, layout):\n    for case, data in norm_profiles[mode][db].items():\n        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)\n\n    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'\n    ax.set_title(f'{db.upper()} | {title_mode}')\n    ax.set_xlabel('Wavelength (µm)')\n    ax.set_ylabel('Optical Depth, τ')\n    ax.grid(alpha=0.25)\n    ax.legend(fontsize=8, loc='upper left')\n    ax.set_ylim(0, 0.1)\n\n    axb = ax.twinx()\n    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)\n    axb.set_ylim(0, 1.1)\n    axb.set_ylabel('F466N transmission (norm.)', color='gray')\n    axb.tick_params(axis='y', colors='gray')\n\nplt.tight_layout()\n\nnorm_table", 647)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.21.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.21.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
layout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]

for ax, (mode, db) in zip(axes.flat, layout):
    for case, data in norm_profiles[mode][db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)

    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'
    ax.set_title(f'{db.upper()} | {title_mode}')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 0.1)

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)
    axb.set_ylim(0, 1.1)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')

plt.tight_layout()

norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
layout = [('fixed_co', 'ocdb'), ('fixed_co', 'bergner'), ('fixed_total_ice', 'ocdb'), ('fixed_total_ice', 'bergner')]

for ax, (mode, db) in zip(axes.flat, layout):
    for case, data in norm_profiles[mode][db].items():
        ax.plot(xarr.value, data['tau_total'], lw=2, label=case)

    title_mode = 'Fixed CO' if mode == 'fixed_co' else 'Fixed total ice'
    ax.set_title(f'{db.upper()} | {title_mode}')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Optical Depth, τ')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, 0.05)

    axb = ax.twinx()
    axb.plot(f466n_wave_um.value, f466n_thr_norm, color='gray', ls='--', lw=1.2, alpha=0.8)
    axb.set_ylim(0, 1.1)
    axb.set_ylabel('F466N transmission (norm.)', color='gray')
    axb.tick_params(axis='y', colors='gray')

plt.tight_layout()

norm_table
#[Out]# <Table length=8>
#[Out]#       mode      database ...   F466N_rel_flux      F466N_delta_mag   
#[Out]#                          ...                                         
#[Out]#      str15        str7   ...      float64              float64       
#[Out]# --------------- -------- ... ------------------ ---------------------
#[Out]#        fixed_co  bergner ...  0.995050158347641   0.00538756711919639
#[Out]#        fixed_co  bergner ... 0.9986772199426089 0.0014371409196070413
#[Out]#        fixed_co     ocdb ...  0.904312954094962   0.10920311984032616
#[Out]#        fixed_co     ocdb ... 0.9844636769100753  0.017000757749483817
#[Out]# fixed_total_ice  bergner ...  0.988956002548695  0.012057573017556542
#[Out]# fixed_total_ice  bergner ... 0.9972469303771616 0.0029932295415767375
#[Out]# fixed_total_ice     ocdb ... 0.4852232057431261    0.7851460930648174
#[Out]# fixed_total_ice     ocdb ...   0.98153770739504  0.020232529006012273
def __jupyter_exec_background__():
    from IPython.display import display
    from threading import Thread
    from traceback import format_exc

    # First send a dummy response to get the display id.
    # Later we'll send the real response with the actual data.
    # And that can happen much later even after the execution completes,
    # as that response will be sent from a bg thread.
    output = display({"application/vnd.vscode.bg.execution.22": ""}, raw=True, display_id=True)

    def do_implementation():
        # Copyright (c) Microsoft Corporation. All rights reserved.
        # Licensed under the MIT License. See LICENSE in the project root
        # for license information.

        # Gotten from ptvsd for supporting the format expected there.
        import sys
        import locale
        from collections import namedtuple
        from importlib.util import find_spec
        import json


        # The pydevd SafeRepr class used in ptvsd/debugpy
        class SafeRepr(object):
            # Can be used to override the encoding from locale.getpreferredencoding()
            locale_preferred_encoding = None

            # Can be used to override the encoding used for sys.stdout.encoding
            sys_stdout_encoding = None

            # String types are truncated to maxstring_outer when at the outer-
            # most level, and truncated to maxstring_inner characters inside
            # collections.
            maxstring_outer = 2**16
            maxstring_inner = 128
            string_types = (str, bytes)
            bytes = bytes
            set_info = (set, "{", "}", False)
            frozenset_info = (frozenset, "frozenset({", "})", False)
            int_types = (int,)
            long_iter_types = (list, tuple, bytearray, range, dict, set, frozenset)

            # Collection types are recursively iterated for each limit in
            # maxcollection.
            maxcollection = (60, 20)

            # Specifies type, prefix string, suffix string, and whether to include a
            # comma if there is only one element. (Using a sequence rather than a
            # mapping because we use isinstance() to determine the matching type.)
            collection_types = [
                (tuple, "(", ")", True),
                (list, "[", "]", False),
                frozenset_info,
                set_info,
            ]
            try:
                from collections import deque

                collection_types.append((deque, "deque([", "])", False))
            except Exception:
                pass

            # type, prefix string, suffix string, item prefix string,
            # item key/value separator, item suffix string
            dict_types = [(dict, "{", "}", "", ": ", "")]
            try:
                from collections import OrderedDict

                dict_types.append((OrderedDict, "OrderedDict([", "])", "(", ", ", ")"))
            except Exception:
                pass

            # All other types are treated identically to strings, but using
            # different limits.
            maxother_outer = 2**16
            maxother_inner = 128

            convert_to_hex = False
            raw_value = False

            def __call__(self, obj):
                """
                :param object obj:
                    The object for which we want a representation.

                :return str:
                    Returns bytes encoded as utf-8 on py2 and str on py3.
                """
                try:
                    return "".join(self._repr(obj, 0))
                except Exception:
                    try:
                        return "An exception was raised: %r" % sys.exc_info()[1]
                    except Exception:
                        return "An exception was raised"

            def _repr(self, obj, level):
                """Returns an iterable of the parts in the final repr string."""

                try:
                    obj_repr = type(obj).__repr__
                except Exception:
                    obj_repr = None

                def has_obj_repr(t):
                    r = t.__repr__
                    try:
                        return obj_repr == r
                    except Exception:
                        return obj_repr is r

                for t, prefix, suffix, comma in self.collection_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_iter(obj, level, prefix, suffix, comma)

                for (
                    t,
                    prefix,
                    suffix,
                    item_prefix,
                    item_sep,
                    item_suffix,
                ) in self.dict_types:  # noqa
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_dict(
                            obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
                        )

                for t in self.string_types:
                    if isinstance(obj, t) and has_obj_repr(t):
                        return self._repr_str(obj, level)

                if self._is_long_iter(obj):
                    return self._repr_long_iter(obj)

                return self._repr_other(obj, level)

            # Determines whether an iterable exceeds the limits set in
            # maxlimits, and is therefore unsafe to repr().
            def _is_long_iter(self, obj, level=0):
                try:
                    # Strings have their own limits (and do not nest). Because
                    # they don't have __iter__ in 2.x, this check goes before
                    # the next one.
                    if isinstance(obj, self.string_types):
                        return len(obj) > self.maxstring_inner

                    # If it's not an iterable (and not a string), it's fine.
                    if not hasattr(obj, "__iter__"):
                        return False

                    # If it's not an instance of these collection types then it
                    # is fine. Note: this is a fix for
                    # https://github.com/Microsoft/ptvsd/issues/406
                    if not isinstance(obj, self.long_iter_types):
                        return False

                    # Iterable is its own iterator - this is a one-off iterable
                    # like generator or enumerate(). We can't really count that,
                    # but repr() for these should not include any elements anyway,
                    # so we can treat it the same as non-iterables.
                    if obj is iter(obj):
                        return False

                    # range reprs fine regardless of length.
                    if isinstance(obj, range):
                        return False

                    # numpy and scipy collections (ndarray etc) have
                    # self-truncating repr, so they're always safe.
                    try:
                        module = type(obj).__module__.partition(".")[0]
                        if module in ("numpy", "scipy"):
                            return False
                    except Exception:
                        pass

                    # Iterables that nest too deep are considered long.
                    if level >= len(self.maxcollection):
                        return True

                    # It is too long if the length exceeds the limit, or any
                    # of its elements are long iterables.
                    if hasattr(obj, "__len__"):
                        try:
                            size = len(obj)
                        except Exception:
                            size = None
                        if size is not None and size > self.maxcollection[level]:
                            return True
                        return any(
                            (self._is_long_iter(item, level + 1) for item in obj)
                        )  # noqa
                    return any(
                        i > self.maxcollection[level] or self._is_long_iter(item, level + 1)
                        for i, item in enumerate(obj)
                    )  # noqa

                except Exception:
                    # If anything breaks, assume the worst case.
                    return True

            def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
                yield prefix

                if level >= len(self.maxcollection):
                    yield "..."
                else:
                    count = self.maxcollection[level]
                    yield_comma = False
                    for item in obj:
                        if yield_comma:
                            yield ", "
                        yield_comma = True

                        count -= 1
                        if count <= 0:
                            yield "..."
                            break

                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    else:
                        if comma_after_single_element:
                            if count == self.maxcollection[level] - 1:
                                yield ","
                yield suffix

            def _repr_long_iter(self, obj):
                try:
                    length = hex(len(obj)) if self.convert_to_hex else len(obj)
                    obj_repr = "<%s, len() = %s>" % (type(obj).__name__, length)
                except Exception:
                    try:
                        obj_repr = "<" + type(obj).__name__ + ">"
                    except Exception:
                        obj_repr = "<no repr available for object>"
                yield obj_repr

            def _repr_dict(
                self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix
            ):
                if not obj:
                    yield prefix + suffix
                    return
                if level >= len(self.maxcollection):
                    yield prefix + "..." + suffix
                    return

                yield prefix

                count = self.maxcollection[level]
                yield_comma = False

                obj_keys = list(obj)

                for key in obj_keys:
                    if yield_comma:
                        yield ", "
                    yield_comma = True

                    count -= 1
                    if count <= 0:
                        yield "..."
                        break

                    yield item_prefix
                    for p in self._repr(key, level + 1):
                        yield p

                    yield item_sep

                    try:
                        item = obj[key]
                    except Exception:
                        yield "<?>"
                    else:
                        for p in self._repr(item, 100 if item is obj else level + 1):
                            yield p
                    yield item_suffix

                yield suffix

            def _repr_str(self, obj, level):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                        else:
                            yield obj
                        return

                    limit_inner = self.maxother_inner
                    limit_outer = self.maxother_outer
                    limit = limit_inner if level > 0 else limit_outer
                    if len(obj) <= limit:
                        # Note that we check the limit before doing the repr (so, the final string
                        # may actually be considerably bigger on some cases, as besides
                        # the additional u, b, ' chars, some chars may be escaped in repr, so
                        # even a single char such as \U0010ffff may end up adding more
                        # chars than expected).
                        yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                        return

                    # Slightly imprecise calculations - we may end up with a string that is
                    # up to 6 characters longer than limit. If you need precise formatting,
                    # you are using the wrong class.
                    left_count, right_count = max(1, int(2 * limit / 3)), max(
                        1, int(limit / 3)
                    )  # noqa

                    # Important: only do repr after slicing to avoid duplicating a byte array that could be
                    # huge.

                    # Note: we don't deal with high surrogates here because we're not dealing with the
                    # repr() of a random object.
                    # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
                    # afterwards, that's ok.

                    # Also, we just show the unicode/string/bytes repr() directly to make clear what the
                    # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
                    # start with b').

                    part1 = obj[:left_count]
                    part1 = repr(part1)
                    part1 = part1[: part1.rindex("'")]  # Remove the last '

                    part2 = obj[-right_count:]
                    part2 = repr(part2)
                    part2 = part2[
                        part2.index("'") + 1 :
                    ]  # Remove the first ' (and possibly u or b).

                    yield part1
                    yield "..."
                    yield part2
                except:
                    # This shouldn't really happen, but let's play it safe.
                    # exception('Error getting string representation to show.')
                    for part in self._repr_obj(
                        obj, level, self.maxother_inner, self.maxother_outer
                    ):
                        yield part

            def _repr_other(self, obj, level):
                return self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer)

            def _repr_obj(self, obj, level, limit_inner, limit_outer):
                try:
                    if self.raw_value:
                        # For raw value retrieval, ignore all limits.
                        if isinstance(obj, bytes):
                            yield obj.decode("latin-1")
                            return

                        try:
                            mv = memoryview(obj)
                        except Exception:
                            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                            return
                        else:
                            # Map bytes to Unicode codepoints with same values.
                            yield mv.tobytes().decode("latin-1")
                            return
                    elif self.convert_to_hex and isinstance(obj, self.int_types):
                        obj_repr = hex(obj)
                    else:
                        obj_repr = repr(obj)
                except Exception:
                    try:
                        obj_repr = object.__repr__(obj)
                    except Exception:
                        try:
                            obj_repr = (
                                "<no repr available for " + type(obj).__name__ + ">"
                            )  # noqa
                        except Exception:
                            obj_repr = "<no repr available for object>"

                limit = limit_inner if level > 0 else limit_outer

                if limit >= len(obj_repr):
                    yield self._convert_to_unicode_or_bytes_repr(obj_repr)
                    return

                # Slightly imprecise calculations - we may end up with a string that is
                # up to 3 characters longer than limit. If you need precise formatting,
                # you are using the wrong class.
                left_count, right_count = max(1, int(2 * limit / 3)), max(
                    1, int(limit / 3)
                )  # noqa

                yield obj_repr[:left_count]
                yield "..."
                yield obj_repr[-right_count:]

            def _convert_to_unicode_or_bytes_repr(self, obj_repr):
                return obj_repr

            def _bytes_as_unicode_if_possible(self, obj_repr):
                # We try to decode with 3 possible encoding (sys.stdout.encoding,
                # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
                # the input, we return the original bytes.
                try_encodings = []
                encoding = self.sys_stdout_encoding or getattr(sys.stdout, "encoding", "")
                if encoding:
                    try_encodings.append(encoding.lower())

                preferred_encoding = (
                    self.locale_preferred_encoding or locale.getpreferredencoding()
                )
                if preferred_encoding:
                    preferred_encoding = preferred_encoding.lower()
                    if preferred_encoding not in try_encodings:
                        try_encodings.append(preferred_encoding)

                if "utf-8" not in try_encodings:
                    try_encodings.append("utf-8")

                for encoding in try_encodings:
                    try:
                        return obj_repr.decode(encoding)
                    except UnicodeDecodeError:
                        pass

                return obj_repr  # Return the original version (in bytes)


        safeRepr = SafeRepr()
        maxStringLength = 1000
        collectionTypes = ["list", "tuple", "set"]
        arrayPageSize = 50

        DisplayOptions = namedtuple("DisplayOptions", ["width", "max_columns"])


        def set_pandas_display_options(display_options=None):
            if find_spec("pandas") is not None:
                try:
                    import pandas as _VSCODE_PD  # type: ignore

                    original_display = DisplayOptions(
                        width=_VSCODE_PD.options.display.width,
                        max_columns=_VSCODE_PD.options.display.max_columns,
                    )

                    if display_options:
                        _VSCODE_PD.options.display.max_columns = display_options.max_columns
                        _VSCODE_PD.options.display.width = display_options.width
                    else:
                        _VSCODE_PD.options.display.max_columns = 100
                        _VSCODE_PD.options.display.width = 1000

                    return original_display
                except ImportError:
                    pass
                finally:
                    del _VSCODE_PD


        def getValue(variable):
            original_display = None
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                original_display = set_pandas_display_options()

            try:
                return safeRepr(variable)
            finally:
                if original_display:
                    set_pandas_display_options(original_display)


        def getPropertyNames(variable):
            props = []
            privateProps = []
            for prop in dir(variable):
                if not prop.startswith("_"):
                    props.append(prop)
                elif not prop.startswith("__"):
                    privateProps.append(prop)
            return props + privateProps


        def getFullType(varType):
            module = ""
            if hasattr(varType, "__module__") and varType.__module__ != "builtins":
                module = varType.__module__ + "."
            if hasattr(varType, "__qualname__"):
                return module + varType.__qualname__
            elif hasattr(varType, "__name__"):
                return module + varType.__name__


        typesToExclude = ["module", "function", "method", "class", "type"]


        def getVariableDescription(variable):
            result = {}

            varType = type(variable)
            result["type"] = getFullType(varType)
            if hasattr(varType, "__mro__"):
                result["interfaces"] = [getFullType(t) for t in varType.__mro__]

            if hasattr(variable, "__len__") and result["type"] in collectionTypes:
                result["count"] = len(variable)

            result["hasNamedChildren"] = hasattr(variable, "__dict__") or isinstance(
                variable, dict
            )

            result["value"] = getValue(variable)
            return result


        def getChildProperty(root, propertyChain):
            try:
                variable = root
                for property in propertyChain:
                    if isinstance(property, int):
                        if hasattr(variable, "__getitem__"):
                            variable = variable[property]
                        elif isinstance(variable, set):
                            variable = list(variable)[property]
                        else:
                            return None
                    elif hasattr(variable, property):
                        variable = getattr(variable, property)
                    elif isinstance(variable, dict) and property in variable:
                        variable = variable[property]
                    else:
                        return None
            except Exception:
                return None

            return variable


        ### Get info on variables at the root level
        def _VSCODE_getVariableDescriptions(varNames):
            variables = [
                {
                    "name": varName,
                    **getVariableDescription(globals()[varName]),
                    "root": varName,
                    "propertyChain": [],
                    "language": "python",
                }
                for varName in varNames
                if varName in globals()
                and type(globals()[varName]).__name__ not in typesToExclude
            ]

            return json.dumps(variables)


        ### Get info on children of a variable reached through the given property chain
        def _VSCODE_getAllChildrenDescriptions(rootVarName, propertyChain, startIndex):
            root = globals()[rootVarName]
            if root is None:
                return []

            parent = root
            if len(propertyChain) > 0:
                parent = getChildProperty(root, propertyChain)

            children = []
            parentInfo = getVariableDescription(parent)
            if "count" in parentInfo:
                if parentInfo["count"] > 0:
                    lastItem = min(parentInfo["count"], startIndex + arrayPageSize)
                    indexRange = range(startIndex, lastItem)
                    children = [
                        {
                            **getVariableDescription(getChildProperty(parent, [i])),
                            "name": str(i),
                            "root": rootVarName,
                            "propertyChain": propertyChain + [i],
                            "language": "python",
                        }
                        for i in indexRange
                    ]
            elif parentInfo["hasNamedChildren"]:
                childrenNames = []
                if hasattr(parent, "__dict__"):
                    childrenNames = getPropertyNames(parent)
                elif isinstance(parent, dict):
                    childrenNames = list(parent.keys())

                children = []
                for prop in childrenNames:
                    child_property = getChildProperty(parent, [prop])
                    if (
                        child_property is not None
                        and type(child_property).__name__ not in typesToExclude
                    ):
                        child = {
                            **getVariableDescription(child_property),
                            "name": prop,
                            "root": rootVarName,
                            "propertyChain": propertyChain + [prop],
                        }
                        children.append(child)

            return json.dumps(children)


        def _VSCODE_getVariableSummary(variable):
            if variable is None:
                return None
            # check if the variable is a dataframe
            if type(variable).__name__ == "DataFrame" and find_spec("pandas") is not None:
                import io

                buffer = io.StringIO()
                variable.info(buf=buffer)
                return json.dumps({"summary": buffer.getvalue()})

            return None


        variables= get_ipython().run_line_magic('who_ls', '')
        return _VSCODE_getVariableDescriptions(variables)

    def bg_main():
        try:
            output.update({"application/vnd.vscode.bg.execution.22.result": do_implementation()}, raw=True)
        except Exception as e:
            output.update({"application/vnd.vscode.bg.execution.22.error": format_exc()}, raw=True)


    Thread(target=bg_main, daemon=True).start()


__jupyter_exec_background__()
del __jupyter_exec_background__
