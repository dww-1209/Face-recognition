import pytest

from face_recognition.domain.errors import (
    CameraDisconnectedError,
    DuplicatePersonError,
    EncodingError,
    FaceRecognitionError,
    LowConfidenceError,
    MultipleFacesError,
    NoFaceError,
    PersonHasNoTemplatesError,
    PersonNotFoundError,
)


def test_all_errors_inherit_base():
    for cls in (
        NoFaceError,
        MultipleFacesError,
        PersonNotFoundError,
        DuplicatePersonError,
        LowConfidenceError,
        PersonHasNoTemplatesError,
        CameraDisconnectedError,
        EncodingError,
    ):
        assert issubclass(cls, FaceRecognitionError)


def test_each_error_has_stable_code():
    assert NoFaceError.code == "NO_FACE"
    assert MultipleFacesError.code == "MULTIPLE_FACES"
    assert PersonNotFoundError.code == "PERSON_NOT_FOUND"
    assert DuplicatePersonError.code == "DUPLICATE_PERSON"
    assert LowConfidenceError.code == "LOW_CONFIDENCE"
    assert PersonHasNoTemplatesError.code == "NO_TEMPLATES"
    assert CameraDisconnectedError.code == "CAMERA_LOST"
    assert EncodingError.code == "ENCODING_ERROR"


def test_multiple_faces_carries_count():
    err = MultipleFacesError(count=3)
    assert err.count == 3


def test_can_be_raised_and_caught_by_base():
    with pytest.raises(FaceRecognitionError):
        raise NoFaceError("没检测到人脸")
