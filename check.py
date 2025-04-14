import urdf2robot as u2r
import spart_functions as ft
import numpy as np

file_name = 'SC_ur10e.urdf'
[robot, robot_key] = u2r.urdf2robot(file_name)
n = robot['n_q']
t0 = np.random.rand(6,1)
tL = np.random.rand(6,n)
P0 = np.random.rand(6,6)
pm = np.random.rand(6,n)
Bi0 = np.random.rand(6,6,n)
Bij = np.random.rand(6,6,n,n)
u0 = np.random.rand(6,1)
um = np.random.rand(n,1)
u0dot = np.random.rand(6,1)
umdot = np.random.rand(n,1)
R0 = np.random.rand(3,3)
r0 = np.random.rand(3,1)
qm = np.random.rand(n,1)

print(robot)
t0dot, tLdot = ft.accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
RJ, RL, rJ, rL, e, g = ft.kinematics(R0, r0, qm, robot)
r_cm = ft.center_of_mass(r0, rL, robot)
Bij, Bi0, P0, pm = ft.diff_kinematics(R0, r0, rL, e, g, robot)
t0_dot, tL_dot = ft.accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
I0, Im = ft.inertia_projection(R0, RL, robot)
M0_tilde, Mm_tilde = ft.mass_composite_body(I0, Im, Bij, Bi0, robot)
H0, H0m, Hm = ft.generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
C0, C0m, Cm0, Cm = ft.convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)