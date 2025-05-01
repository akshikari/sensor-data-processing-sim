# Python

## NumPy/SciPy

### [NumPy](https://numpy.org/doc/stable/)

- [$$numpy.einsum](https://numpy.org/doc/2.1/reference/generated/numpy.einsum.html)

  - Related [video](https://www.youtube.com/watch?v=CLrTj7D2fLM)

- [$$numpy.stack](https://numpy.org/doc/stable/reference/generated/numpy.stack.html)

## AWS packages

### boto3

- Seems to be the more common standard, though `s3fs` exists.
- Started out with `s3fs`, but no stubs package/file exists yet so switched to `boto3`
- `boto3` also has `moto3` to mock AWS resources for testing purposes

### Pytest

- Honestly my first time doing a more extensive application of Pytest and I am a fan.
- New features I've learned about:
  - `fixtures`: More or less constants that you can set up at different levels of scope (per function, per test suite, etc.) to be reused and reduce resource usage
  - `caplog`: An optional parameter you can pass to test functions that will capture the logs. Handy if your test is looking for some kind of logs output from the function
