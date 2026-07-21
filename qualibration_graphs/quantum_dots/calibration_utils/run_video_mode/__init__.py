from .video_mode_specific_parameters import VideoModeCommonParameters


def create_video_mode(*args, **kwargs):
    from .video_mode_utils import create_video_mode as _create_video_mode

    return _create_video_mode(*args, **kwargs)


__all__ = ["create_video_mode", "VideoModeCommonParameters"]
