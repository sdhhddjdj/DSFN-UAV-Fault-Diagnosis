"""Compatibility entry point for the revised training CLI."""
import sys
from reproduce import main
if __name__=='__main__':
    sys.argv.insert(1,'train')
    main()
