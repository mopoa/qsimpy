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
gate rzz(theta) a,b
{
  cx a,b;
  u3(0,0,theta) b;
  cx a,b;
}

// relative-phase CCX

// relative-phase 3-controlled X gate

// 3-controlled X gate

// 3-controlled sqrt(X) gate, this equals the C3X gate 
// where the CU1 rotations are -pi/8 not -pi/4

// 4-controlled X gate 

qubit[8] q;
bit[8] meas;
h q[0];
h q[1];
h q[2];
rzz(-1.8524208893612883) q[0],q[2];
h q[3];
rzz(-1.8524208893612883) q[2],q[3];
rx(10.069641922878436) q[2];
h q[4];
h q[5];
rzz(-1.8524208893612883) q[1],q[5];
rzz(-1.8524208893612883) q[4],q[5];
rx(10.069641922878436) q[5];
h q[6];
rzz(-1.8524208893612883) q[0],q[6];
rx(10.069641922878436) q[0];
rzz(5.638770534540546) q[0],q[2];
rzz(-1.8524208893612883) q[4],q[6];
rx(10.069641922878436) q[4];
rx(10.069641922878436) q[6];
rzz(5.638770534540546) q[0],q[6];
rx(-1.289448413559494) q[0];
h q[7];
rzz(-1.8524208893612883) q[1],q[7];
rx(10.069641922878436) q[1];
rzz(5.638770534540546) q[1],q[5];
rzz(-1.8524208893612883) q[3],q[7];
rx(10.069641922878436) q[3];
rzz(5.638770534540546) q[2],q[3];
rx(-1.289448413559494) q[2];
rzz(5.638770534540546) q[4],q[5];
rzz(5.638770534540546) q[4],q[6];
rx(-1.289448413559494) q[4];
rx(-1.289448413559494) q[5];
rx(-1.289448413559494) q[6];
rx(10.069641922878436) q[7];
rzz(5.638770534540546) q[1],q[7];
rx(-1.289448413559494) q[1];
rzz(5.638770534540546) q[3],q[7];
rx(-1.289448413559494) q[3];
rx(-1.289448413559494) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];