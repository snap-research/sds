import copy
import sys
import time
import traceback
from typing import BinaryIO, Callable, List, Union

import av
from PIL import Image


def identity_transform(x):
    return x


class FramesBuffer:
    def __init__(self, max_size: int, max_times_difference: float):
        """
        Defines a frame buffer holding frames and their timestamps
        :param max_size: The maximum number of frames to keep in the buffer
        :param max_times_difference: The maximum time difference between the first and last frame in the buffer
        """

        self.frames = []
        self.max_size = max_size
        self.max_times_difference = max_times_difference
        self.min_cache_valid = False
        self.max_cache_valid = False
        self.min_cache = None
        self.max_cache = None

    def push(self, frame, timestamp: float):
        while len(self.frames) >= self.max_size or (
            self.max_times_difference is not None
            and len(self.frames) >= 1
            and (
                self.max_timestamp() - self.min_timestamp() > self.max_times_difference
            )
        ):
            self.frames.pop(0)
            self.invalidate_cache()
        self.frames.append((frame, timestamp))
        self.invalidate_cache()

    def invalidate_cache(self):
        self.min_cache_valid = False
        self.max_cache_valid = False

    def min_timestamp(self) -> float:
        if self.min_cache_valid:
            return self.min_cache

        min_time = float("inf")
        for frame, timestamp in self.frames:
            if timestamp < min_time:
                min_time = timestamp

        self.min_cache_valid = True
        self.min_cache = min_time
        return min_time

    def max_timestamp(self) -> float:
        if self.max_cache_valid:
            return self.max_cache

        max_time = float("-inf")
        for frame, timestamp in self.frames:
            if timestamp > max_time:
                max_time = timestamp

        self.max_cache_valid = True
        self.max_cache = max_time

        return max_time

    def get_closest_frame(
        self, timestamp: float, tolerance: float
    ) -> Image.Image | None:
        """
        Retrieves the closest frame in the buffer within the specified tolerance.
        :param timestamp: The target timestamp to find the closest frame for.
        :param tolerance: The maximum allowed difference between the target timestamp and the frame's timestamp.
        :return: The closest frame within the specified tolerance, or None if no such frame exists.
        """

        closest_frame = None
        closest_time = float("inf")

        # Finds the closest frame to the target timestamp
        for frame, frame_timestamp in self.frames:
            if (
                abs(frame_timestamp - timestamp) <= tolerance
                and abs(frame_timestamp - timestamp) < closest_time
            ):
                closest_frame = frame
                closest_time = abs(frame_timestamp - timestamp)

        return closest_frame, closest_time


class VideoDecoder:
    """Utility class for decoding videos"""

    def __init__(
        self,
        file: Union[str, BinaryIO],
        default_thread_type: str = None,
        enable_frame_parallel_decoding: bool = False,
        gop_size_hint_forward: int = 10,
        gop_size_hint_backward: int = 5,
        max_buffer_size_frames: int = 3,
        max_buffer_size_time: float = None,
    ):
        """:param file: The video file to decode
        :param default_thread_type: The default thread type to use for decoding the video.
                                    The use of threading may reduce performance when decoding single, randomly-accessed frames
        :param enable_frame_parallel_decoding: Whether to use multiple threads for decoding multiple frames
                                               May cause excessive thread creation if video-level parallelism is used already
        :param gop_size_hint_forward: The number of frames that is acceptable to decode forward in time to reach the frame of interest rather than performing a seek operation
                                      Seek is an expensive operation, so it may be beneficial to set the parameter to a value larger than the gop size if the gop size is small, eg. 5 frames
        :param gop_size_hint_backward: The number of frames that we will go backward in time if the seek overshoots the target

        """
        self.file = file
        if isinstance(file, str):
            self.file = open(file, "rb")

        # Some video metadata may break pyav for unknown reasons (https://github.com/PyAV-Org/PyAV/issues/629)
        # We ignore errors in metadata loading. Even with this option it seems the other metadata remain readable
        self.container = av.open(file, mode="r", metadata_errors="ignore")

        self.video_stream = self.container.streams.video[0]
        self.framerate = float(
            self.video_stream.guessed_rate,
        )  # use guessed_rate which is more robust than codec_context.framerate which only looks at a few frames
        self.frame_duration = 1 / self.framerate

        # The default threading value used by the video stream
        self.default_video_stream_thread_type = default_thread_type
        if self.default_video_stream_thread_type is None:
            self.default_video_stream_thread_type = self.video_stream.thread_type
        self.video_stream.thread_type = self.default_video_stream_thread_type

        # Enabling this slows down decoding for a single frame, but can accelerate decoding of multiple frames
        # Linked to the use of
        # video_stream.thread_type = "AUTO"
        self.enable_frame_parallel_decoding = enable_frame_parallel_decoding

        # Seeks only if at least a number of frames greater than the GOP size is present between the current and next frame
        # Better to decode more subsequent frames than risk overshooting with a seek
        self.min_seek_time_interval = gop_size_hint_forward / self.framerate

        # If the seek overshoots the target, we go backwards in time of a common GOP size
        self.backward_search_time_step = gop_size_hint_backward / self.framerate

        # Dimensions for the frames buffer for sequential reading
        self.max_buffer_size_frames = max_buffer_size_frames
        self.max_buffer_size_time = max_buffer_size_time

    def close(self):
        self.container.close()
        self.file.close()

    def apply_transformation(
        self,
        frame: Image.Image,
        transform: Callable,
    ) -> Image.Image:
        """Applies a transformation to a frame"""
        if transform is None:
            transform = identity_transform

        try:
            transformed_frame = transform(frame)
        except Exception as e:
            print_message = f"An exception occurred in VideoDecoder while transforming a frame in {self.file}, "
            exception_message = str(e)
            exception_trace_string = traceback.format_exc()
            print_message += f"Exception message: {exception_message}\n"
            print_message += f"Exception trace:\n{exception_trace_string}"
            print(print_message, file=sys.stderr, flush=True)
            raise e

        return transformed_frame

    def maybe_use_identity(self, transform: Callable) -> Callable:
        """If the transform is None, use the identity transform"""
        if transform is None:
            return identity_transform
        return transform

    def decode_all_frames(
        self,
        transform: Callable = None,
    ) -> List[Image.Image]:
        """Decodes all the frames of the video
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: List of PIL images corresponding to the decoded frames
        """
        transform = self.maybe_use_identity(transform)

        decoded_images = []
        for frame in self.container.decode(video=0):
            frame = frame.to_image()
            transformed_frame = self.apply_transformation(frame, transform)
            decoded_images.append(transformed_frame)
        return decoded_images

    def decode_frame_at_index(
        self,
        index: int,
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> Image.Image:
        """Extracts a frame at the given index
        :param index: The index of the frame to read. The index is expressed assuming constant framerate and no missing frames
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: PIL image with the decoded frame
        """
        timestamp = index / self.framerate
        return self.decode_frame_at_time(
            timestamp,
            frame_seek_timeout_sec,
            transform,
        )

    def decode_frame_at_time(
        self,
        timestamp: int,
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> Image.Image:
        """Extracts a frame at the given timestamp
        :param timestamp: Timestamp of the frame to read
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: PIL image with the decoded frame
        """
        results = self.decode_frames_at_times(
            [timestamp],
            frame_seek_timeout_sec,
            transform,
        )
        return results[0]

    def decode_frames_at_indexes(
        self,
        indexes: List[int],
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> List[Image.Image]:
        """Extracts the frames corresponding to the given indexes
        :param indexes: The indexes of the frames to read. The indexes are expressed assuming constant framerate and no missing frames
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: List of PIL images corresponding to the decoded frames
        """
        timestamps = [index / self.framerate for index in indexes]
        return self.decode_frames_at_times(
            timestamps,
            frame_seek_timeout_sec,
            transform,
        )

    def decode_frames_at_times(
        self,
        timestamps: List[float],
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> List[Image.Image]:
        """Extracts the frames corresponding to the given timestamps
        :param timestamps: The timestamps of the frames to read
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: List of PIL images corresponding to the decoded frames
        """
        # If more than one frame needs to be decoded and frame parallelism is allowed, we enable it
        # Frame parallelism introduces delay if used to read only a limited number of frames
        if (
            not self.video_stream.codec_context.is_open
        ):  # We can change the thread type only of the stream is not open already
            if len(timestamps) > 1 and self.enable_frame_parallel_decoding:
                self.video_stream.thread_type = "AUTO"
            else:
                self.video_stream.thread_type = self.default_video_stream_thread_type

        # Sorts the timestamps
        timestamp_with_index = [
            (timestamp, index) for index, timestamp in enumerate(timestamps)
        ]
        timestamp_with_index.sort(key=lambda x: x[0])

        decoded_images = []

        # Iterator that iterates the files being decoded. Necessary to keep the iterator open between different iterations as calling self.container.decode(video=0) multiple times can generate EOF errors
        decoding_iterator = None
        last_decoded_timestamp = float("-inf")
        for (
            current_target_timestamp,
            original_order_index,
        ) in timestamp_with_index:
            timeout_timer = time.time()

            # Special case where timestamps corresponding to the same frame are requested. Recycles the last decoded frame
            if (
                abs(last_decoded_timestamp - current_target_timestamp)
                <= self.frame_duration / 2 + 0.001
            ):
                frame_to_clone = decoded_images[-1][0]
                last_frame_copy = copy.deepcopy(
                    frame_to_clone,
                )  # Uses deepcopy as we do not know the type returned by the transformation
                decoded_images.append((last_frame_copy, original_order_index))
                continue

            current_search_start_time = current_target_timestamp
            # Loop to adjust imprecise seek location
            found = False
            while not found:
                # Seeks only if at least a number of frames greater than a large GOP size is present between the current and next frame
                # Or if the frame we need to read is behind us
                # last_decoded_timestamp >= current_target_timestamp needed to reinitialize the iterator since the same frame cannot be read more than once without seeking again and reopining the iterator
                if (
                    current_target_timestamp - last_decoded_timestamp
                    >= self.min_seek_time_interval
                    or last_decoded_timestamp >= current_target_timestamp
                ):
                    seek_offset = round(
                        current_search_start_time / self.video_stream.time_base,
                    )
                    self.container.seek(
                        seek_offset,
                        backward=True,
                        any_frame=False,
                        stream=self.video_stream,
                    )
                    decoding_iterator = (
                        None  # need to open a new iterator for decoding at seek
                    )

                try:
                    # Reads frames
                    loop_break_exit = False
                    if decoding_iterator is None:
                        decoding_iterator = iter(
                            self.container.decode(video=0),
                        )
                        minimum_time_encountered_frame_from_last_seek = float(
                            "inf",
                        )  # Detects whether a frame is missing if the target is between min and current and yet we have not found the frame
                        previous_decoded_timestamp = None
                    while True:
                        current_timeout_timer = time.time()
                        if (
                            current_timeout_timer - timeout_timer
                            > frame_seek_timeout_sec
                        ):
                            raise Exception(
                                f"Timeout of {frame_seek_timeout_sec}s reached while decoding frame at time {current_target_timestamp} in the current video {self.file}.",
                            )

                        frame = next(decoding_iterator)
                        # for frame in self.container.decode(video=0):
                        last_decoded_timestamp = (
                            frame.time
                        )  # The timestamp of the currently read frame
                        # Frames are unordered, raise an exception
                        if (
                            previous_decoded_timestamp is not None
                            and last_decoded_timestamp < previous_decoded_timestamp
                        ):
                            raise Exception(
                                "Frames in the video are unordered, the file may be corrupted. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering. This is not supported",
                            )
                        previous_decoded_timestamp = last_decoded_timestamp
                        minimum_time_encountered_frame_from_last_seek = min(
                            minimum_time_encountered_frame_from_last_seek,
                            last_decoded_timestamp,
                        )

                        # We found a frame that is closer than half the distance between each frame, so it is the closest to the target we can get
                        # Timestamps may be rounded, so we add a small epsilon to account for expanded intervals between frames due to rounding
                        if (
                            abs(
                                last_decoded_timestamp - current_target_timestamp,
                            )
                            <= self.frame_duration / 2 + 0.001
                        ):
                            found = True
                            loop_break_exit = True
                            break
                        # If we scanned from a frame that was lower than the target to a frame that is greater than the target without finding the target, then it means that the target is missing
                        if (
                            last_decoded_timestamp > current_target_timestamp
                            and minimum_time_encountered_frame_from_last_seek
                            < current_target_timestamp
                        ):
                            raise Exception(
                                f"Could not find the frame at time {current_target_timestamp} between the seek timestamp {minimum_time_encountered_frame_from_last_seek} and the current timestamp {last_decoded_timestamp} in the current video {self.file}. The video frame may be missing of video frames may be unordered. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering.",
                            )
                        # The seek overshoot the target
                        if last_decoded_timestamp > current_target_timestamp:
                            loop_break_exit = True
                            break
                except StopIteration:
                    # If reading the last frame, the right interval covered by the last frame is not frame_duration / 2, but frame duration
                    if (
                        current_target_timestamp > last_decoded_timestamp
                        and abs(
                            last_decoded_timestamp - current_target_timestamp,
                        )
                        <= self.frame_duration + 0.001
                    ):
                        found = True
                        loop_break_exit = True
                    else:
                        continue
                except Exception as e:
                    print_message = f"An exception occurred in VideoDecoder while decoding frames in {self.file}, "
                    exception_message = str(e)
                    exception_trace_string = traceback.format_exc()
                    print_message += f"Exception message: {exception_message}\n"
                    print_message += f"Exception trace:\n{exception_trace_string}"
                    print(print_message, file=sys.stderr, flush=True)

                # Frame not found
                if not found and (
                    current_search_start_time == 0 or not loop_break_exit
                ):
                    raise Exception(
                        f"Could not find frame at time {current_target_timestamp} in the current video {self.file}",
                    )

                # If we overshoot the target with the seek, we go backwards in time for the next seek
                current_search_start_time -= self.backward_search_time_step
                current_search_start_time = max(current_search_start_time, 0)

            frame_pil = frame.to_image()
            transformed_frame = self.apply_transformation(frame_pil, transform)
            decoded_images.append((transformed_frame, original_order_index))

        # Sorts back the results in the initial order
        decoded_images.sort(key=lambda x: x[1])
        decoded_images = [image for image, _ in decoded_images]

        return decoded_images

    def _exact_search_from_zero(self, target_timestep: float):
        """
        Assuming the container is already seeked to the beginning of the video, this function will search for the preceding frame closest to the target timestamp
        Unfortunately the seed function will usually return the successive one and not the preceding
        """

        # Checks which is the first video frame
        current_iterator = iter(
            self.container.decode(video=0),
        )
        first_video_frame = next(current_iterator)
        first_video_timestamp = first_video_frame.time

        if (target_timestep - first_video_timestamp) < self.min_seek_time_interval:
            return first_video_frame, current_iterator

        # Seeks forward if the sought frame is too far away
        decoded_time = 0
        seek_offset = round(
            target_timestep / self.video_stream.time_base,
        )
        self.container.seek(
            seek_offset,
            backward=True,
            any_frame=False,
            stream=self.video_stream,
        )

        current_iterator = iter(
            self.container.decode(video=0),
        )
        current_frame = next(current_iterator)
        decoded_time = current_frame.time

        # Seek successfully returned a preceding frame
        if decoded_time < target_timestep:
            return current_frame, current_iterator

        # Seeks backward until we arrive at a preceding frame
        seek_target = target_timestep
        while True:
            seek_target = max(0, seek_target - self.backward_search_time_step)
            seek_offset = round(
                seek_target / self.video_stream.time_base,
            )
            self.container.seek(
                seek_offset,
                backward=True,
                any_frame=False,
                stream=self.video_stream,
            )

            current_iterator = iter(
                self.container.decode(video=0),
            )
            current_frame = next(current_iterator)
            decoded_time = current_frame.time

            if seek_target == 0 or decoded_time < target_timestep:
                # We have reached the beginning of the video or we have found a frame before the target
                return current_frame, current_iterator

    def decode_frames_at_indexes_approx(
        self,
        indexes: List[int],
        max_tolerance: float = None,
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> List[Image.Image]:
        """Extracts the frames corresponding to the given indexes using flexible frames search. Useful for videos with missing frames
        :param indexes: The indexes of the frames to read. The indexes are expressed assuming constant framerate and no missing frames
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: List of PIL images corresponding to the decoded frames
        """
        timestamps = [index / self.framerate for index in indexes]
        return self.decode_frames_at_times_approx(
            timestamps,
            max_tolerance,
            frame_seek_timeout_sec,
            transform,
        )

    def decode_frames_at_times_approx(
        self,
        timestamps: List[float],
        max_tolerance: float = None,
        frame_seek_timeout_sec: float = 10.0,
        transform: Callable = None,
    ) -> List[Image.Image]:
        """Extracts the frames corresponding to the given timestamps using flexible frames search. Useful for videos with missing frames
        :param timestamps: The timestamps of the frames to read
        :param max_tolerance: The maximum tolerance in seconds for which a requested timestamp can be associated to a decoded frame
        :param frame_seek_timeout_sec: The maximum time to wait for a frame to be decoded. If the timeout is reached, the decoding is aborted and an exception is raised.
                                       Acts as a safeguard against unforeseen corruptions of the underlying media file
        :param transform: Optional transformation to apply to each function
        :return: List of PIL images corresponding to the decoded frames
        """
        if max_tolerance is None:
            # max_tolerance = 0.051  # 50ms -> Tolerate 1 lost frame at 20fps
            max_tolerance = 0.102  # 102ms

        # If more than one frame needs to be decoded and frame parallelism is allowed, we enable it
        # Frame parallelism introduces delay if used to read only a limited number of frames
        if (
            not self.video_stream.codec_context.is_open
        ):  # We can change the thread type only of the stream is not open already
            if len(timestamps) > 1 and self.enable_frame_parallel_decoding:
                self.video_stream.thread_type = "AUTO"
            else:
                self.video_stream.thread_type = self.default_video_stream_thread_type

        # Sorts the timestamps
        timestamp_with_index = [
            (timestamp, index) for index, timestamp in enumerate(timestamps)
        ]
        timestamp_with_index.sort(key=lambda x: x[0])

        decoded_images = []
        frame_buffer = FramesBuffer(
            max_size=self.max_buffer_size_frames,
            max_times_difference=self.max_buffer_size_time,
        )  # Buffer to hold the frames and their timestamps

        # Iterator that iterates the files being decoded. Necessary to keep the iterator open between different iterations as calling self.container.decode(video=0) multiple times can generate EOF errors
        decoding_iterator = None

        try:
            all_decoded_timestamp = []
            first_timestamp = timestamp_with_index[0][0]

            firts_frame, decoding_iterator = self._exact_search_from_zero(
                first_timestamp
            )

            file_finished = False

            all_decoded_timestamp.append(firts_frame.time)

            current_decoded_timestamp = (
                firts_frame.time
            )  # The timestamp of the currently read frame
            last_decoded_timestamp = current_decoded_timestamp
            frame_buffer.push(firts_frame, current_decoded_timestamp)

            for current_target_timestamp, original_order_index in timestamp_with_index:
                timeout_timer_start = time.time()

                while True:
                    timeout_timer_end = time.time()
                    # Checks for timeouts
                    if timeout_timer_end - timeout_timer_start > frame_seek_timeout_sec:
                        raise Exception(
                            f"Timeout of {frame_seek_timeout_sec}s reached while decoding frame at time {current_target_timestamp} in the current video {self.file}.",
                        )

                    if (
                        current_target_timestamp <= frame_buffer.max_timestamp()
                        or file_finished
                    ):
                        retrieved_frame, _ = frame_buffer.get_closest_frame(
                            current_target_timestamp, max_tolerance
                        )
                        # If we can't find the frame, then it's missing
                        if retrieved_frame is None:
                            raise Exception(
                                f"Could not find the frame at time {current_target_timestamp}. The video frame may be missing frames or video frames may be unordered. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering.",
                            )
                        # Otherwise we can use the frame
                        else:
                            break

                    # We need to read ahead to find the frame
                    try:
                        decoded_frame = next(decoding_iterator)
                        last_decoded_timestamp = current_decoded_timestamp
                        current_decoded_timestamp = decoded_frame.time
                        all_decoded_timestamp.append(current_decoded_timestamp)
                        if last_decoded_timestamp > current_decoded_timestamp:
                            raise Exception(
                                f"Frames in the video are unordered. Found successive timestamps to be {last_decoded_timestamp} and {current_decoded_timestamp}. When opening the file the decoder may have logged 'CTTS invalid' indicating invalid mapping between decoding and presentation timestamps, leading to lack of ordering. This is not supported",
                            )
                    # We finished the file, loop again and match
                    except StopIteration:
                        file_finished = True
                        continue
                    current_decoded_timestamp = decoded_frame.time
                    frame_buffer.push(decoded_frame, current_decoded_timestamp)

                # Here retrieved frame contains the matched frame
                frame_pil = retrieved_frame.to_image()
                transformed_frame = self.apply_transformation(frame_pil, transform)
                decoded_images.append((transformed_frame, original_order_index))

            # Sorts back the results in the initial order
            decoded_images.sort(key=lambda x: x[1])
            decoded_images = [image for image, _ in decoded_images]

            return decoded_images

        except Exception as e:
            print_message = f"An exception occurred in VideoDecoder while decoding frames in {self.file}, "
            exception_message = str(e)
            exception_trace_string = traceback.format_exc()
            print_message += f"Exception message: {exception_message}\n"
            print_message += f"Exception trace:\n{exception_trace_string}\n"
            print_message += f"Exception decoded timesteps: {all_decoded_timestamp}\n"
            print(print_message, file=sys.stderr, flush=True)
            raise e


def main():
    test_video = "/home/willi/dl/animation/video-generation-3d/data/benchmarking/real_data/vp9_crf_48_g5.webm"
    decoder = VideoDecoder(test_video)

    timstamps = [0.1 + 0.5 * i for i in range(30)]
    start = time.time()
    images = decoder.decode_frames_at_times(timstamps)
    end = time.time()
    print(
        f"Decoding {len(timstamps)} frames took {end - start} seconds, fps: {len(timstamps) / (end - start):.3f}",
    )

    for image in images:
        image.show()

    print("Done")


if __name__ == "__main__":
    main()
