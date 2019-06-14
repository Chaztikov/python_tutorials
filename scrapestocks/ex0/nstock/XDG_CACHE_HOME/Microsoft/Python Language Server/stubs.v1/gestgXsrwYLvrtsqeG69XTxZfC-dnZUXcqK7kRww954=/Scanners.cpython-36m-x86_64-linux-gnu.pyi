import __future__ as _mod___future__
import builtins as _mod_builtins

BOL = 'bol'
EOF = 'eof'
EOL = 'eol'
NOT_FOUND = _mod_builtins.object()
class Scanner(_mod_builtins.object):
    "\n    A Scanner is used to read tokens from a stream of characters\n    using the token set specified by a Plex.Lexicon.\n\n    Constructor:\n\n      Scanner(lexicon, stream, name = '')\n\n        See the docstring of the __init__ method for details.\n\n    Methods:\n\n      See the docstrings of the individual methods for more\n      information.\n\n      read() --> (value, text)\n        Reads the next lexical token from the stream.\n\n      position() --> (name, line, col)\n        Returns the position of the last token read using the\n        read() method.\n\n      begin(state_name)\n        Causes scanner to change state.\n\n      produce(value [, text])\n        Causes return of a token value to the caller of the\n        Scanner.\n\n    "
    __class__ = Scanner
    __dict__ = {}
    def __init__(self, lexicon, stream, name, initial_pos):
        "\n        Scanner(lexicon, stream, name = '')\n\n          |lexicon| is a Plex.Lexicon instance specifying the lexical tokens\n          to be recognised.\n\n          |stream| can be a file object or anything which implements a\n          compatible read() method.\n\n          |name| is optional, and may be the name of the file being\n          scanned or any other identifying string.\n        "
        pass
    
    @classmethod
    def __init_subclass__(cls):
        'This method is called when a class is subclassed.\n\nThe default implementation does nothing. It may be\noverridden to extend subclasses.\n'
        return None
    
    __module__ = 'Cython.Plex.Scanners'
    @classmethod
    def __subclasshook__(cls, subclass):
        'Abstract classes can override this to customize issubclass().\n\nThis is invoked early on by abc.ABCMeta.__subclasscheck__().\nIt should return True, False or NotImplemented.  If it returns\nNotImplemented, the normal algorithm is used.  Otherwise, it\noverrides the normal algorithm (and the outcome is cached).\n'
        return False
    
    @property
    def __weakref__(self):
        'list of weak references to the object (if defined)'
        pass
    
    def begin(self, state_name):
        'Set the current state of the scanner to the named state.'
        pass
    
    def eof(self):
        '\n        Override this method if you want something to be done at\n        end of file.\n        '
        pass
    
    def get_position(self):
        'Python accessible wrapper around position(), only for error reporting.\n        '
        pass
    
    def next_char(self):
        pass
    
    def position(self):
        '\n        Return a tuple (name, line, col) representing the location of\n        the last token read using the read() method. |name| is the\n        name that was provided to the Scanner constructor; |line|\n        is the line number in the stream (1-based); |col| is the\n        position within the line of the first character of the token\n        (0-based).\n        '
        pass
    
    def produce(self, value, text):
        '\n        Called from an action procedure, causes |value| to be returned\n        as the token value from read(). If |text| is supplied, it is\n        returned in place of the scanned text.\n\n        produce() can be called more than once during a single call to an action\n        procedure, in which case the tokens are queued up and returned one\n        at a time by subsequent calls to read(), until the queue is empty,\n        whereupon scanning resumes.\n        '
        pass
    
    def read(self):
        "\n        Read the next lexical token from the stream and return a\n        tuple (value, text), where |value| is the value associated with\n        the token as specified by the Lexicon, and |text| is the actual\n        string read from the stream. Returns (None, '') on end of file.\n        "
        pass
    
    def run_machine_inlined(self):
        '\n        Inlined version of run_machine for speed.\n        '
        pass
    
    def scan_a_token(self):
        "\n        Read the next input sequence recognised by the machine\n        and return (text, action). Returns ('', None) on end of\n        file.\n        "
        pass
    

__builtins__ = {}
__cached__ = '/usr/lib/python3/dist-packages/Cython/Plex/__pycache__/Scanners.cpython-37.pyc'
__doc__ = None
__file__ = '/usr/lib/python3/dist-packages/Cython/Plex/Scanners.py'
__name__ = 'Cython.Plex.Scanners'
__package__ = 'Cython.Plex'
absolute_import = _mod___future__._Feature()
