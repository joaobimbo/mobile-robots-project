#!/bin/bash
ustreamer --device /dev/video0 --format=JPEG --encoder=HW --workers=3 --desired-fps=15 --quality 60 --flip-vertical 1 --flip-horizontal 1 --host=0.0.0.0 --port=12345