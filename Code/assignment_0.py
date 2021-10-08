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

    parser.add_argument(
        '--no-plot',
        dest='plot',
        action='store_false',
        help='Add flag to avoid plotting',
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
            format="%(levelname)s: %(asctime)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if args.input_file is not None:
            recording = compute_values.Recording(
                file_path=args.input_file,
            )
            logging.info("## Compute Values for Pre-Recording Reference ##")
            compute_values.compute_values(
                calibrated_recording=recording.calibrated_pre,
                samplerate=recording.samplerate,
                plot=args.plot,
            )
            logging.info("## Compute Values for Post-Recording Reference ##")
            compute_values.compute_values(
                calibrated_recording=recording.calibrated_post,
                samplerate=recording.samplerate,
                plot=args.plot,
            )

        elif args.run_all:
            compute_values.main(plot=args.plot)

