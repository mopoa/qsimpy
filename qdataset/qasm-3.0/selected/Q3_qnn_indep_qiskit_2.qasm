// Generated from qBraid v0.7.1
OPENQASM 3.0;
include 'stdgates.inc';
// Generated from qiskit v0.24.0 file qiskit/qasm/libs/qelib1.inc

// QASM 3 equivalent definitions of qelib1.inc gates

// idle gate (identity) with length gamma*sqglen
 
// generic single qubit gate

// inverse sqrt(X)

// controlled-sqrt(X)

// controlled phase rotation 

// controlled-U

// two-qubit XX rotation
// gate rxx(theta) a, b
// {
//   u3(pi/2, theta, 0) a;
//   h b;
//   cx a,b;
//   u1(-theta) b;
//   cx a,b;
//   h b;
//   ******** why type error ********
//   // u2(-pi, pi-theta) b;
// }

// two-qubit ZZ rotation

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[2] q;
bit[2] meas;
u2(2.0,-pi) q[0];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
u2(2.0,-pi) q[0];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
ry(0.525921735635995) q[0];
ry(0.938837580580377) q[1];
cx q[0],q[1];
ry(0.556818227561703) q[0];
ry(0.894383585795336) q[1];
barrier q[0],q[1];
meas[0] = measure q[0];
meas[1] = measure q[1];