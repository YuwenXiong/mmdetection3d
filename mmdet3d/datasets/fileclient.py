import mmcv
from mmcv.fileio.file_client import (
    HardDiskBackend,
    CephBackend,
    MemcachedBackend,
    LmdbBackend,
    PetrelBackend,
    HTTPBackend,
    BaseStorageBackend,
)
from contextlib import contextmanager

import fsspec
import copy
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing import Any, Generator, Iterator, Optional, Tuple, Union
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import tempfile

import s3fs
from io import BytesIO
from s3fs import S3FileSystem
import boto3
from pathlib import Path
from pyarrow.fs import FileSystem


class RelativeFileSystemProtocol(str, Enum):
    FILE = "file"  # local file system
    S3 = "s3"


def ensure_trailing_slash(path: str) -> str:
    """Returns a copy of the path which always has a trailing slash.

    Examples:
        >>> ensure_trailing_slash("foo")
        'foo/'

        >>> ensure_trailing_slash("/some/path")
        '/some/path/'

        >>> ensure_trailing_slash("/")
        '/'

        >>> ensure_trailing_slash("")
        '/'

    """
    # We don't use ``os.path`` since we are working with URIs, so we don't actually want OS-specific separators.
    if path.endswith("/"):
        return path
    else:
        return path + "/"


def get_original_cwd():
    """Returns the original program working directory, in a Hydra-aware manner."""
    return os.getcwd()


def remove_leading_slash(path: str) -> str:
    """Removes the path's leading slash(es), if it/they exists.

    Examples:
        >>> remove_leading_slash("/foo")
        'foo'

        >>> remove_leading_slash("//foo")
        'foo'

        >>> remove_leading_slash("foo")
        'foo'

        >>> remove_leading_slash("foo/bar")
        'foo/bar'

        >>> remove_leading_slash("foo/bar/")
        'foo/bar/'
    """
    return re.sub(r"^\.?/+", "", path)


def remove_leading_slash_fsspec(entry: Union[str, dict]) -> Union[str, dict]:
    """Removes a leading slash for a directory entry produced by fsspec.

    In non-detailed mode, this is just a string so we use ``remove_leading_slash``. In detailed mode, ``fsspec``
    returns a dict with a ``name`` key, so we returned an updated dict.
    """
    if isinstance(entry, str):
        return remove_leading_slash(entry)
    else:
        new_entry = copy.deepcopy(entry)
        new_entry["name"] = remove_leading_slash(new_entry["name"])
        return new_entry


def _strip_base_path(entry: Union[str, dict], base_path: str) -> Union[str, dict]:
    """Removes the base path from a fsspec directory entry.

    Example:
        >>> _strip_base_path({"name": "foo/bar"}, "foo")
        {'name': '/bar'}

    """
    assert not base_path.startswith("/")
    if isinstance(entry, str):
        return entry.replace(base_path, "")
    else:
        new_entry = copy.deepcopy(entry)
        new_entry["name"] = new_entry["name"].replace(base_path, "")
        return new_entry


class RelativeFileSystem(fsspec.AbstractFileSystem):
    def __init__(self, base_path: str, base_fs: fsspec.AbstractFileSystem):
        """Implements a filesystem relative to an arbitrary base path using ``fsspec``.

        Warning:
            Using this with S3 with multiprocessing can cause the underlying s3fs library calls to hang. Possible
            solutions are either using ``multiprocessing.set_start_method('forkserver')`` in your program before
            forking, or using ``boto3`` directly if the S3 operations are simple enough.

        This is equivalent to PyArrow's SubTreeFileSystem, just using ``fsspec``.  This was implemented since we need
        more flexibility than what fsspec's FSMap gives us.

        The main use case is cleanly exposing a specific sub-tree of a filesystem to an application, without passing
        absolute paths for everything. For instance, an application could have a root data URI in its config, then
        define all other paths (metadata, raw data, output logs) relative to this, and perform IO through
        ``RelativeFileSystem``, which allows transparent switching between NFS, S3, and other storage solutions.
        Multiple roots are definitely possible, e.g., one for all inputs, and one for the outputs.

        For example, a simulation asset library contains many asset geometry files, asset metadata files, etc., but
        they are all relative to a common root folder, which can be present either locally, on NFS, or in S3. However,
        applications shouldn't care about where exactly the root is, so this helper allows them to just deal with the
        subset of a filesystem that they care about.

        Please see ``AssetIO`` and the asset generation code base for usage examples of this class.
        """
        super().__init__()
        if not isinstance(base_fs, fsspec.AbstractFileSystem):
            raise TypeError(f"Please pass an instance of fsspec.AbstractFileSystem. Got {type(base_fs)=}.")

        self._base_path = ensure_trailing_slash(base_path)
        self._base_fs = base_fs

        self._implement_relative_methods()

    @staticmethod
    def from_base_uri(base_uri: str, filesystem_kwargs: Optional[dict] = None) -> "RelativeFileSystem":
        """Initializes a filesystem relative to the provided ``base_uri``.

        Args:
            base_uri:           The URI to be used as the filesystem root. All RelativeFileSystem paths will be relative
                                to this. The URI path must be absolute. If no scheme is provided, ``file://`` is
                                assumed.
            filesystem_kwargs:  Optional arguments to be passed to the specific filesystem implementation being used.

        For instance:

            >>> from waabi.common.data.relative_fs import RelativeFileSystem
            >>> fs = RelativeFileSystem.from_base_uri("file:///tmp/xyz/")

        would create an filesystem object which would access local files relative to ``/tmp/xyz``.
        ``fs.open("file.txt")`` would therefore attempt to load the local path ``/tmp/xyz/file.txt``. The same also
        works for S3 URIs such as:

            >>> fs = RelativeFileSystem.from_base_uri("s3://my-bucket/my-prefix")

        where again ``fs.open("file.txt")`` would attempt to load ``s3://my-bucket/my-prefix/bar.ply``.
        Any `<fsspec protocol> https://filesystem-spec.readthedocs.io/en/stable/api.html#built-in-implementations`_,
        such as ``file``, ``s3``, ``hdfs``, etc. should be compatible.

        Leading slashes to paths passed to RelativeFileSystem methods are ignored.
        """
        parsed_uri = urlparse(base_uri)
        if parsed_uri.scheme:
            scheme = parsed_uri.scheme
        else:
            scheme = "file"

        # Drops the scheme (first tuple element) since we pass it separately to the constructor
        schemeless_uri_str = urlunparse(("", *parsed_uri[1:]))

        if scheme == "file" and not os.path.isabs(schemeless_uri_str):
            # The user passed a relative path to a URI (e.g., "./some/dir"), so we make it absolute.
            schemeless_uri_str = os.path.abspath(os.path.join(get_original_cwd(), base_uri))

        if filesystem_kwargs is None:
            filesystem_kwargs = {}

        if scheme == "s3" and "use_listings_cache" not in filesystem_kwargs:
            # When using S3 we disable the listings cache by default, to avoid issues where it causes subsequent
            # file existence checks to fail.
            #
            # TODO(andrei): Disable this override if we identify a more specific root cause, or if a future version of
            #               s3fs eliminates the need for this.
            filesystem_kwargs["use_listings_cache"] = False

        filesystem = fsspec.filesystem(scheme, **filesystem_kwargs)
        base_path = schemeless_uri_str

        return RelativeFileSystem(base_path=base_path, base_fs=filesystem)

    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        """Alias for ``base_fs`` for compatibility with the fsspec 'CachingFileSystem``."""
        return self.base_fs

    @property
    def base_fs(self) -> fsspec.AbstractFileSystem:
        """The raw underlying filesystem. Meant for advanced use only."""
        return self._base_fs

    @property
    def base_path(self) -> str:
        return self._base_path

    @property
    def base_uri(self) -> str:
        """Base uri with the protocol prefix (e.g., s3://xxx)"""
        return f"{self.fs_protocol.value}:{self._base_path}"

    @property
    def fs_protocol(self) -> RelativeFileSystemProtocol:
        """Protocol prefix for the raw base uri, e.g., 'file' or 's3'.

        The difference from the default inherited protocol attribute from `fsspec.AbstractFileSystem` is that
        this property is always a custom enum defined by RelativeFileSystemProtocol, while the default inherited
        protocol can be a string, or a list (e.g., s3fs.S3FileSystem.protocol).
        """
        base_fs_protocol = self._base_fs.protocol
        if isinstance(base_fs_protocol, str):
            try:
                RelativeFileSystemProtocol(base_fs_protocol)
            except ValueError as err:
                if "enum for value" in str(err):
                    raise RuntimeError(f"Unsupported RelativeFileSystemProtocol {base_fs_protocol}") from err
                else:
                    raise
            return RelativeFileSystemProtocol(base_fs_protocol)
        if isinstance(self._base_fs, s3fs.S3FileSystem):
            # s3fs.S3FileSystem.protocol is ["s3", "s3a"], we return "s3"
            return RelativeFileSystemProtocol.S3
        raise ValueError(
            f"Unsupported protocol for file system {type(self._base_fs)} with default protocol {base_fs_protocol}"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, type(self)) and other._base_fs == self._base_fs and other._base_path == other._base_path
        )

    def _implement_relative_methods(self):
        """Implements ``fsspec.AbstractFileSystem`` by generating methods to call it with the relative ``base_path``.

        This bypasses the need to write a large amount of boilerplate code.
        """

        def call_with_prefix(fun, prefix):
            """Adds a fixed string prefix to the first argument of the wrapped function/method ``fun``.

            Used to automatically make methods like "ls" use the right prefix. Only works with functions whose first
            argument is meant to be a path.
            """

            def wrapped(*args, **kwargs):
                if len(args) > 0:
                    path, *other_args = args
                else:
                    path, other_args = "", []

                prefixed_path = prefix + remove_leading_slash(path)
                return fun(prefixed_path, *other_args, **kwargs)

            return wrapped

        def call_with_prefix_2_paths(fun, prefix):
            """Adds a fixed string prefix to the first & second argument of the wrapped function/method ``fun``.

            Used to automatically make methods like "cp" use the right prefix. Only works with functions whose first
            two arguments are meant to be a paths.

            The first path argument is a relative path, whereas the second argument is of arbitrary format, to permit
            for operations from the relative path to other paths
            """

            def wrapped(*args, **kwargs):
                if len(args) > 1:
                    path_1, path_2, *other_args = args
                elif len(args) > 0:
                    path_1, path_2, other_args = args[0], "", []
                else:
                    path_1, path_2, other_args = "", "", []

                prefixed_path_1 = prefix + remove_leading_slash(path_1)
                return fun(prefixed_path_1, path_2, *other_args, **kwargs)

            return wrapped

        method_names_to_wrap = [
            "cat",
            "created",
            "exists",
            "info",
            "isdir",
            "isfile",
            "makedir",
            "makedirs",
            "mkdir",
            "mkdirs",
            "modified",
            "open",
            "rm",
            "size",
            "stat",
            "touch",
        ]
        for method_name in method_names_to_wrap:
            original_method = getattr(self._base_fs, method_name)
            setattr(self, method_name, call_with_prefix(original_method, self._base_path))

        method_names_to_wrap_2_paths = ["cp", "mv", "rename"]
        for method_name in method_names_to_wrap_2_paths:
            original_method = getattr(self._base_fs, method_name)
            setattr(self, method_name, call_with_prefix_2_paths(original_method, self._base_path))

    def ls(self, relative_path: str = "/", detail=False, **kwargs):
        """Returns a listing of relative paths w.r.t. the base directory."""
        results = self._base_fs.ls(self._base_path + remove_leading_slash(relative_path), detail, **kwargs)

        # NOTE: To fix bug where for a s3 uri "s3://waabi/example", the base_path is resolved to "//waabi" with leading
        #       slashes whereas the results are resolved to ["waabi/example"].
        #       The inconsistency results in returning full paths rather than relative paths
        base_path = remove_leading_slash(self._base_path)
        results = [remove_leading_slash_fsspec(result) for result in results]

        # We remove the base path since by definitions users of our code should not care about full paths.
        return [remove_leading_slash_fsspec(_strip_base_path(entry, base_path)) for entry in results]

    def du(self, relative_path: str = "/", total=True, maxdepth=None, **kwargs):
        """Returns size of relative path w.r.t. the base directory."""
        return self._base_fs.du(self._base_path + remove_leading_slash(relative_path), total, maxdepth, **kwargs)

    @property
    def is_s3_versioned(self) -> bool:
        return isinstance(self.base_fs, s3fs.S3FileSystem) and self.base_fs.version_aware

    def get_versions(self, relative_path: str = "/") -> List[Dict[str, Any]]:
        """Return the JSON format version data available for a file"""
        if not isinstance(self.base_fs, s3fs.S3FileSystem):
            raise TypeError(
                "`get_versions` only supported for versioned S3FileSystem,"
                f"but current file system is {type(self.base_fs)}"
            )
        if not self.base_fs.version_aware:
            raise AttributeError(
                'Provided S3FileSystem is not version_aware, during creation set kwargs: `{"version_aware":True})`'
            )
        return self.base_fs.object_version_info(self._base_path + remove_leading_slash(relative_path))


def s3_reader(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    buf = bytearray(response["ContentLength"])
    view = memoryview(buf)
    pos = 0
    while True:
        chunk = response["Body"].read(67108864)
        if len(chunk) == 0:
            break
        view[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    return view


@mmcv.FileClient.register_backend("awss3", prefixes="awss3")
class S3Backend(BaseStorageBackend):
    """Petrel storage backend (for internal use).

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Default: True.

    Examples:
        >>> filepath1 = 's3://path/of/file'
        >>> filepath2 = 'cluster-name:s3://path/of/file'
        >>> client = PetrelBackend()
        >>> client.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self, base_uri):
        self._client = RelativeFileSystem.from_base_uri(base_uri)
        self.base_uri_len = len(base_uri)
        fs, basepath = FileSystem.from_uri(base_uri)
        # self.basepath = Path(basepath)
        # self.bucket = str(self.basepath.parents[1])
        self.bucket = base_uri.split('//')[1].split('/')[0]
        self.bucket_len = len(f"s3://{self.bucket}/")

    def get(self, filepath: Union[str, Path], mode="rb") -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            memoryview: A memory view of expected bytes object to avoid
                copying. The memoryview object can be converted to bytes by
                ``value_buf.tobytes()``.
        """
        # if filepath.startswith(self._client.base_uri):
        #     filepath = filepath[self.base_uri_len :]

        # s3 = S3FileSystem()
        # with s3.open(filepath, mode) as f:
        # # value = self._client.open(filepath, mode)
        # # return BytesIO(value.read())
        #     value_buf = memoryview(f.read())

        # print(self.bucket, filepath[self.bucket_len:])
        s3_client = boto3.client("s3")

        for i in range(10):
            try:
                value_buf = s3_reader(s3_client, self.bucket, filepath[self.bucket_len :])
                break
            except:
                pass

        return value_buf

    def get_text(self, filepath: Union[str, Path], encoding: str = "utf-8") -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return str(self.get(filepath, mode="r"), encoding=encoding)

    # def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
    #     """Save data to a given ``filepath``.

    #     Args:
    #         obj (bytes): Data to be saved.
    #         filepath (str or Path): Path to write data.
    #     """
    #     filepath = self._map_path(filepath)
    #     filepath = self._format_path(filepath)
    #     self._client.put(filepath, obj)

    # def put_text(self,
    #              obj: str,
    #              filepath: Union[str, Path],
    #              encoding: str = 'utf-8') -> None:
    #     """Save data to a given ``filepath``.

    #     Args:
    #         obj (str): Data to be written.
    #         filepath (str or Path): Path to write data.
    #         encoding (str): The encoding format used to encode the ``obj``.
    #             Default: 'utf-8'.
    #     """
    #     self.put(bytes(obj, encoding=encoding), filepath)

    # def exists(self, filepath: Union[str, Path]) -> bool:
    #     """Check whether a file path exists.

    #     Args:
    #         filepath (str or Path): Path to be checked whether exists.

    #     Returns:
    #         bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
    #     """
    #     if not (has_method(self._client, 'contains')
    #             and has_method(self._client, 'isdir')):
    #         raise NotImplementedError(
    #             'Current version of Petrel Python SDK has not supported '
    #             'the `contains` and `isdir` methods, please use a higher'
    #             'version or dev branch instead.')

    #     filepath = self._map_path(filepath)
    #     filepath = self._format_path(filepath)
    #     return self._client.contains(filepath) or self._client.isdir(filepath)

    # def isdir(self, filepath: Union[str, Path]) -> bool:
    #     """Check whether a file path is a directory.

    #     Args:
    #         filepath (str or Path): Path to be checked whether it is a
    #             directory.

    #     Returns:
    #         bool: Return ``True`` if ``filepath`` points to a directory,
    #         ``False`` otherwise.
    #     """
    #     if not has_method(self._client, 'isdir'):
    #         raise NotImplementedError(
    #             'Current version of Petrel Python SDK has not supported '
    #             'the `isdir` method, please use a higher version or dev'
    #             ' branch instead.')

    #     filepath = self._map_path(filepath)
    #     filepath = self._format_path(filepath)
    #     return self._client.isdir(filepath)

    # def isfile(self, filepath: Union[str, Path]) -> bool:
    #     """Check whether a file path is a file.

    #     Args:
    #         filepath (str or Path): Path to be checked whether it is a file.

    #     Returns:
    #         bool: Return ``True`` if ``filepath`` points to a file, ``False``
    #         otherwise.
    #     """
    #     if not has_method(self._client, 'contains'):
    #         raise NotImplementedError(
    #             'Current version of Petrel Python SDK has not supported '
    #             'the `contains` method, please use a higher version or '
    #             'dev branch instead.')

    #     filepath = self._map_path(filepath)
    #     filepath = self._format_path(filepath)
    #     return self._client.contains(filepath)

    # def join_path(self, filepath: Union[str, Path],
    #               *filepaths: Union[str, Path]) -> str:
    #     """Concatenate all file paths.

    #     Args:
    #         filepath (str or Path): Path to be concatenated.

    #     Returns:
    #         str: The result after concatenation.
    #     """
    #     filepath = self._format_path(self._map_path(filepath))
    #     if filepath.endswith('/'):
    #         filepath = filepath[:-1]
    #     formatted_paths = [filepath]
    #     for path in filepaths:
    #         formatted_paths.append(self._format_path(self._map_path(path)))
    #     return '/'.join(formatted_paths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` and return a temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str | Path): Download a file from ``filepath``.

        Examples:
            >>> client = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with client.get_local_path('s3://path/of/your/file') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one temporary path.
        """
        # if filepath.startswith(self._client.base_uri):
        #     filepath = filepath[self.base_uri_len :]

        assert self._client.isfile(filepath[self.base_uri_len :])
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    # def list_dir_or_file(self,
    #                      dir_path: Union[str, Path],
    #                      list_dir: bool = True,
    #                      list_file: bool = True,
    #                      suffix: Optional[Union[str, Tuple[str]]] = None,
    #                      recursive: bool = False) -> Iterator[str]:
    #     """Scan a directory to find the interested directories or files in
    #     arbitrary order.

    #     Note:
    #         Petrel has no concept of directories but it simulates the directory
    #         hierarchy in the filesystem through public prefixes. In addition,
    #         if the returned path ends with '/', it means the path is a public
    #         prefix which is a logical directory.

    #     Note:
    #         :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
    #         In addition, the returned path of directory will not contains the
    #         suffix '/' which is consistent with other backends.

    #     Args:
    #         dir_path (str | Path): Path of the directory.
    #         list_dir (bool): List the directories. Default: True.
    #         list_file (bool): List the path of files. Default: True.
    #         suffix (str or tuple[str], optional):  File suffix
    #             that we are interested in. Default: None.
    #         recursive (bool): If set to True, recursively scan the
    #             directory. Default: False.

    #     Yields:
    #         Iterable[str]: A relative path to ``dir_path``.
    #     """
    #     if not has_method(self._client, 'list'):
    #         raise NotImplementedError(
    #             'Current version of Petrel Python SDK has not supported '
    #             'the `list` method, please use a higher version or dev'
    #             ' branch instead.')

    #     dir_path = self._map_path(dir_path)
    #     dir_path = self._format_path(dir_path)
    #     if list_dir and suffix is not None:
    #         raise TypeError(
    #             '`list_dir` should be False when `suffix` is not None')

    #     if (suffix is not None) and not isinstance(suffix, (str, tuple)):
    #         raise TypeError('`suffix` must be a string or tuple of strings')

    #     # Petrel's simulated directory hierarchy assumes that directory paths
    #     # should end with `/`
    #     if not dir_path.endswith('/'):
    #         dir_path += '/'

    #     root = dir_path

    #     def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
    #                           recursive):
    #         for path in self._client.list(dir_path):
    #             # the `self.isdir` is not used here to determine whether path
    #             # is a directory, because `self.isdir` relies on
    #             # `self._client.list`
    #             if path.endswith('/'):  # a directory path
    #                 next_dir_path = self.join_path(dir_path, path)
    #                 if list_dir:
    #                     # get the relative path and exclude the last
    #                     # character '/'
    #                     rel_dir = next_dir_path[len(root):-1]
    #                     yield rel_dir
    #                 if recursive:
    #                     yield from _list_dir_or_file(next_dir_path, list_dir,
    #                                                  list_file, suffix,
    #                                                  recursive)
    #             else:  # a file path
    #                 absolute_path = self.join_path(dir_path, path)
    #                 rel_path = absolute_path[len(root):]
    #                 if (suffix is None
    #                         or rel_path.endswith(suffix)) and list_file:
    #                     yield rel_path

    #     return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
    #                              recursive)
