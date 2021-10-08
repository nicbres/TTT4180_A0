#!./venv/bin/python
import argparse
import logging

import compute_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run code for TTT4180 Technical Acoustics Assignment 0.')

    parser.add_argument(
        '--input', '-i',
        dest='input_file',
        action='store',
        default=None,
        help='The path to the input file',
    )

    parser.add_argument(
        '--all', '-a',
        dest='run_all',
        action='store_true',
        default=None,
        help='Run code for the 30min sound file. Must be available at hardcoded paths',
    )

    parser.add_argument(
        '--debug', '-d',
        dest='debug',
        action='store_true',
        help='Change logging level to debug',
    )

    parser.add_argument(
        '--log-file',
        dest='log_file',
        action='store',
        default=None,
        help='Add path to store log to file instead of stdout',
    )

    args = parser.parse_args()

    if args.run_all is None and args.input_file is None:
        parser.print_help()
    else:
        # Setup logging
        logging.basicConfig(
            filename=args.log_file,
            filemode='w' if args.log_file else None,
            level=logging.DEBUG if args.debug else logging.INFO,
        )

        if args.input_file is not None:
            # hardcoded calibration factors computed with reference recordings
            calibration_factor_pre = 1.6831914497821373e-09
            calibration_factor_post = 1.8531190247712699e-09

        elif args.run_all:
            compute_values.main()

        


