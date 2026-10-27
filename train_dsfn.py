"""Revised DSFN model API and explicit training entry point."""
from dsfn.models import build_model,build_base_model,wrap_gtd
if __name__=='__main__':
    import sys
    from reproduce import main
    sys.argv[1:1]=['train','--family','dsfn']
    main()
