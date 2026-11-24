"""启动入口：python main.py [选项]，查看选项：python main.py -h"""
import sys

from mimir.cli import main

if __name__ == "__main__":
    sys.exit(main())
