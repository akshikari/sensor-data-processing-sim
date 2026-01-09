# Glossary

Defined terms to be used in describing the domain objects and their functionalities.
Most will be defined in layman's terms just enough to make the code and logic make
sense.

## Robot

This entire project centers around simulating the output of an accelerometer
sensor as if it were on a quadruped (four-legged) robot. For the purpose
of this project, we only go as far as providing a unique identifier for
such a hypothetical robot mainly for demo purposes and nothing more.

## IMU

Each of these robots is equipped with a hypothetical IMU, or
Inertial Measurement Unit. These IMUs typically house various sensors all
geared towards measuring and reporting various metrics such as specific
force, angular rate, and orientation which are all measured via sensors.
The sensors that typically come with an IMU are accelerometers, gyroscopes, and magnetometers.

## Specific Force

Simply the force per unit mass on a body. This is the acceleration measured
by an accelerometer in $m/s^2$, and is the non-gravitational acceleration
acting on the body.
The formula in this context is $f = a_{\text{linear}} - g$ where $g$ is the
gravity vector. For example, if the robot was standing still, the accelerometer
would be outputting $+9.81 m/s^2$ because this is what the accelerometer
"feels" on the body's frame as the ground pushing up on the body to fight
the force of gravity. $f = 0 - (-9.81 m/s^2) = +9.81 m/s^2$

## Proper Acceleration

The acceleration experienced by a body relative to free fall. This is the
_magnitude_ of specific force expressed as a vector in the body frame.

## Gravity Vector ($`g`$)

The vector representing the acceleration a body feels due to the forces
of gravity. This is along the z-axis, and is represented by the NumPy
vector `[0, 0, -9.81]`

## Body Frame

Coordinate system that is "attached" to the robot/sensor. This is what the
accelerometer measures acceleration along the axes of this coordinate plane.
In the accelerometer model, we first calculate acceleration in the world frame,
then convert it to the body frame to properly simulate accelerometer outputs.

## World Frame

A fixed coordinate system representing the external environment. 3 axes:
x = forward/back, y = left/right, z = up/down, or `[x, y, z]`. Acceleration
is calculated along each of these axes before applying a Rotation Matrix to
convert the values to the body frame. We calculate acceleration in the world
frame first because that is naturally how we describe motion through space.
It's the trajectory of a body relative to a fixed environment.

## Rotation Matrix

A 3x3 orthogonal matrix where each column represents the body frame's axes in
the world matrix. This is a linear transformation used to represent how the
body frame is oriented relative to the world frame. It is constructed from
the robot's roll, pitch, and yaw angles:

$R_{bw} = R_x(roll)R_y(pitch)R_z(yaw)$

The transpose of this matrix can be do the inverse and express the world frame
relative to the body frame. This is what is used to express the calculated
acceleration values as well as the gravity vector in the robot's body frame
to more closely simulate how an IMU would output the acceleration values.

## Roll $`\phi`$

The value representing the angle of rotation about the x-axis. Analagous to
tilting your head side to side. Follows a sinusoidal pattern to mimic the
natural sway of a robot as it walks. A positive value represents the right
side of the robot tilting downward.

## Pitch $`\theta`$

The value representing the angle of rotation about the y-axis. Analagous to
nodding your head up and down. Follows a sinusoidal pattern to mimic the
natural sway of a robot as it walks. A positive value represents the nose
of the robot tilting upward.

This along with roll changes the orientation of the gravity vector in the
body frame, hence why they are important to calculate.

## Yaw $`\psi`$

The value representing the angle of rotation about the z-axis. Analagous to
turning your head left and right. For now, this is a constant 0 in the
accelerometer model as we assume the robot is walking in a straight line
for simplicity's sake. A positive value represents the robot turning left,
following the right-hand rule.

This along with roll and pitch are known as the Euler Angles and represent
a body's orientation in 3D space.

## Sway

The side-to-side lateral motion of the robot as it shifts weight from one leg
to another while walking. This sway motion repeats every two steps, therefore
it's relationship can be defined as `gait_frequency / 2` in the accelerometer
model. This value is used in calculating the lateral acceleration (along the
y-axis).

## Bounce

The up-and-down motion of the robot's body as each foot makes contact with and
pushes off the ground. This is used in calculating the main vertical oscillation
that the IMU will detect

## Gait

Frequency at which the the robot is taking steps. If the robot is taking 2 steps
per second, its gait frequency is 2 Hz This configurable parameter is used in
calculating the Angular Frequency $\omega_{\text{gait}}$ which is just how fast
the oscillations occur in radians per second. Finally this angular frequency
value is used in calculating the roll and pitch of the robot for a given time
$t$.

## Angular Rate/Velocity

The rate of change of the angular position over time, and can be interpreted as
how fast something is rotating along some axis, ($\omega = \frac{d\theta}{dt}$).
Often this is directly measured by a sensor like a gyroscope. This is unused
in the current model, but it's useful to know that the 1st derivative of the
Simplar Harmonic Motion formula (SHM) will give us the formula for angular
velocity. Note that this is slightly different from angular frequency in that
angular velocity is a vector which means it describes both the magnitude _and_
direction of the velocity.

## Simple Harmonic Motion (SHM)

Simple Harmonic Motion is a type of periodic motion where a system oscillates
smoothly around an equilibrium position and experiences a restoring acceleration
proportional to its displacement. The motion is always sinusoidal in time and
can be fully described by three parameters: amplitude, angular frequnecy, and
phase. For a given time $t$, to find the position of a body following SHM:

$x(t) = A \cdot sin(\omega t + \phi)$

Differentiating the above twice gives us the acceleration at time $t$:

$a(t) = A\omega^2 \cdot sin(\omega t + \phi)$

This calculation for acceleration is the core assumption of our model. Given
ideal walking conditions (straight-line path, steady gait, stable speed), the
quadruped body exhibits spring-like oscillations. The following all follow
the SHM periodic motion:

- Vertical bounce
- Lateral sway
- Roll and pitch (rotational oscillations)

Thee above are all independent SHM signals, and the accelerations describing
these motions are the main output of our accelerometer model. This allows us
to simulate plausible IMU behavior without a full dynamics engine.

## Amplitude ($A$)

The maximum displacement of an oscillating value from its equilibrium position.
The $A$ in the SHM formula

## Phase ($\phi$)

The point in the cycle at which an oscillation begins at time $t = 0$. The
$\phi$ in the SHM formula.

## Angular Frequency ($\omega$)

How quickly an oscillation completes cycles in radians per second. The $\omega$
in the SHM formula.

## Orientation

How a body is rotated in 3D space relative to some reference frame (in our case
the world frame).

## Sensor

Any device that detects and measures a physical property and converts it into a
signal meaningful to our system. Accelerometers, gyroscopes. magnetometers.

## Accelerometer

A sensor that measures specific force, or the acceleration excluding gravity,
expressed in the body frame.

## Gyroscope

Measures angular rate (or rotation speed) around each body axis in radians/second.
Answers the question of "how fast is the robot turning right now?". Measures the
changes in orientation over time.

## Magnetometers

Measures the Earth's magnetic field vector in the sensor's body frame in order to
estimate absolute yaw heading (compass direction).

## Sampling Rate

In this context, it is the rate at which the data is produced, measured in Hz.

## Monotonic Clock

A timer that only moves forward, is independent of the system clock the process
is running on, and measures the duration or amount of time passed and not the
actual time of day.

## Wall Clock

The real-world, timezone-aware calendar time. The "system time".

## Effective Time

This is the robot's "perceived time" based on its internal clock. In the
absence of anomalies this time is in sync with the wall clock/system time.
These are the timestamps emitted by the simulated IMU, and internally this
is the time used to calculate the various sensor data points to best mimic
a real-world IMU. Once time-related anomalies are introduced, this
effective time will differ from the wall clock time.

!> [!NOTE]

> The below terms are more to do with real-time data processing and are mostly
> abstracted away from the end user, but are still good to know.

## Stream

A continuous sequence of data points produced over time. Can be thought of as
a pipelines, with one input and one or more outputs. This library produces
streams of sensor data with a timestamp associated with each data point.

## Stream Session

An _instance_ of a stream. Using inputs of _start time_, _sensor state_, and the
_parameters_ used as inputs to generate the stream of sensor data.

## Sensor State

Used interchangeably with _Stream State_. An object used to maintain variables
pertinent to defining the current state of a stream session. Optionally, can
be used to resume a stream session that was previously paused.

<!--TODO: Fill this in later-->:

## Producer

The process that is creating the data being pushed to the stream. In our case
it would be the sensor data generators.

## Consumer

The process that takes in and possibly processes the data coming out of the
stream.

## Backpressure

What occurs when a consumer begins to lag behind the producer. This results
in data "building up" in the pipeline and can result in increased latency and
data loss/quality issues. Typically this is handled by signalling the producer
to slow down or pause until the consumer(s) can catch up.

## Pipeline

A pipeline is a sequence of 0 or more processing stages that the data flows through.
