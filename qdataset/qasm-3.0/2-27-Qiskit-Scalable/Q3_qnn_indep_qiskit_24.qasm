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

qubit[24] q;
bit[24] meas;
u2(2.0,-pi) q[0];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
u2(2.0,-pi) q[4];
cx q[0],q[4];
p(9.17283818781952) q[4];
cx q[0],q[4];
cx q[1],q[4];
p(9.17283818781952) q[4];
cx q[1],q[4];
cx q[2],q[4];
p(9.17283818781952) q[4];
cx q[2],q[4];
cx q[3],q[4];
p(9.17283818781952) q[4];
cx q[3],q[4];
u2(2.0,-pi) q[5];
cx q[0],q[5];
p(9.17283818781952) q[5];
cx q[0],q[5];
cx q[1],q[5];
p(9.17283818781952) q[5];
cx q[1],q[5];
cx q[2],q[5];
p(9.17283818781952) q[5];
cx q[2],q[5];
cx q[3],q[5];
p(9.17283818781952) q[5];
cx q[3],q[5];
cx q[4],q[5];
p(9.17283818781952) q[5];
cx q[4],q[5];
u2(2.0,-pi) q[6];
cx q[0],q[6];
p(9.17283818781952) q[6];
cx q[0],q[6];
cx q[1],q[6];
p(9.17283818781952) q[6];
cx q[1],q[6];
cx q[2],q[6];
p(9.17283818781952) q[6];
cx q[2],q[6];
cx q[3],q[6];
p(9.17283818781952) q[6];
cx q[3],q[6];
cx q[4],q[6];
p(9.17283818781952) q[6];
cx q[4],q[6];
cx q[5],q[6];
p(9.17283818781952) q[6];
cx q[5],q[6];
u2(2.0,-pi) q[7];
cx q[0],q[7];
p(9.17283818781952) q[7];
cx q[0],q[7];
cx q[1],q[7];
p(9.17283818781952) q[7];
cx q[1],q[7];
cx q[2],q[7];
p(9.17283818781952) q[7];
cx q[2],q[7];
cx q[3],q[7];
p(9.17283818781952) q[7];
cx q[3],q[7];
cx q[4],q[7];
p(9.17283818781952) q[7];
cx q[4],q[7];
cx q[5],q[7];
p(9.17283818781952) q[7];
cx q[5],q[7];
cx q[6],q[7];
p(9.17283818781952) q[7];
cx q[6],q[7];
u2(2.0,-pi) q[8];
cx q[0],q[8];
p(9.17283818781952) q[8];
cx q[0],q[8];
cx q[1],q[8];
p(9.17283818781952) q[8];
cx q[1],q[8];
cx q[2],q[8];
p(9.17283818781952) q[8];
cx q[2],q[8];
cx q[3],q[8];
p(9.17283818781952) q[8];
cx q[3],q[8];
cx q[4],q[8];
p(9.17283818781952) q[8];
cx q[4],q[8];
cx q[5],q[8];
p(9.17283818781952) q[8];
cx q[5],q[8];
cx q[6],q[8];
p(9.17283818781952) q[8];
cx q[6],q[8];
cx q[7],q[8];
p(9.17283818781952) q[8];
cx q[7],q[8];
u2(2.0,-pi) q[9];
cx q[0],q[9];
p(9.17283818781952) q[9];
cx q[0],q[9];
cx q[1],q[9];
p(9.17283818781952) q[9];
cx q[1],q[9];
cx q[2],q[9];
p(9.17283818781952) q[9];
cx q[2],q[9];
cx q[3],q[9];
p(9.17283818781952) q[9];
cx q[3],q[9];
cx q[4],q[9];
p(9.17283818781952) q[9];
cx q[4],q[9];
cx q[5],q[9];
p(9.17283818781952) q[9];
cx q[5],q[9];
cx q[6],q[9];
p(9.17283818781952) q[9];
cx q[6],q[9];
cx q[7],q[9];
p(9.17283818781952) q[9];
cx q[7],q[9];
cx q[8],q[9];
p(9.17283818781952) q[9];
cx q[8],q[9];
u2(2.0,-pi) q[10];
cx q[0],q[10];
p(9.17283818781952) q[10];
cx q[0],q[10];
cx q[1],q[10];
p(9.17283818781952) q[10];
cx q[1],q[10];
cx q[2],q[10];
p(9.17283818781952) q[10];
cx q[2],q[10];
cx q[3],q[10];
p(9.17283818781952) q[10];
cx q[3],q[10];
cx q[4],q[10];
p(9.17283818781952) q[10];
cx q[4],q[10];
cx q[5],q[10];
p(9.17283818781952) q[10];
cx q[5],q[10];
cx q[6],q[10];
p(9.17283818781952) q[10];
cx q[6],q[10];
cx q[7],q[10];
p(9.17283818781952) q[10];
cx q[7],q[10];
cx q[8],q[10];
p(9.17283818781952) q[10];
cx q[8],q[10];
cx q[9],q[10];
p(9.17283818781952) q[10];
cx q[9],q[10];
u2(2.0,-pi) q[11];
cx q[0],q[11];
p(9.17283818781952) q[11];
cx q[0],q[11];
cx q[1],q[11];
p(9.17283818781952) q[11];
cx q[1],q[11];
cx q[2],q[11];
p(9.17283818781952) q[11];
cx q[2],q[11];
cx q[3],q[11];
p(9.17283818781952) q[11];
cx q[3],q[11];
cx q[4],q[11];
p(9.17283818781952) q[11];
cx q[4],q[11];
cx q[5],q[11];
p(9.17283818781952) q[11];
cx q[5],q[11];
cx q[6],q[11];
p(9.17283818781952) q[11];
cx q[6],q[11];
cx q[7],q[11];
p(9.17283818781952) q[11];
cx q[7],q[11];
cx q[8],q[11];
p(9.17283818781952) q[11];
cx q[8],q[11];
cx q[9],q[11];
p(9.17283818781952) q[11];
cx q[9],q[11];
cx q[10],q[11];
p(9.17283818781952) q[11];
cx q[10],q[11];
u2(2.0,-pi) q[12];
cx q[0],q[12];
p(9.17283818781952) q[12];
cx q[0],q[12];
cx q[1],q[12];
p(9.17283818781952) q[12];
cx q[1],q[12];
cx q[2],q[12];
p(9.17283818781952) q[12];
cx q[2],q[12];
cx q[3],q[12];
p(9.17283818781952) q[12];
cx q[3],q[12];
cx q[4],q[12];
p(9.17283818781952) q[12];
cx q[4],q[12];
cx q[5],q[12];
p(9.17283818781952) q[12];
cx q[5],q[12];
cx q[6],q[12];
p(9.17283818781952) q[12];
cx q[6],q[12];
cx q[7],q[12];
p(9.17283818781952) q[12];
cx q[7],q[12];
cx q[8],q[12];
p(9.17283818781952) q[12];
cx q[8],q[12];
cx q[9],q[12];
p(9.17283818781952) q[12];
cx q[9],q[12];
cx q[10],q[12];
p(9.17283818781952) q[12];
cx q[10],q[12];
cx q[11],q[12];
p(9.17283818781952) q[12];
cx q[11],q[12];
u2(2.0,-pi) q[13];
cx q[0],q[13];
p(9.17283818781952) q[13];
cx q[0],q[13];
cx q[1],q[13];
p(9.17283818781952) q[13];
cx q[1],q[13];
cx q[2],q[13];
p(9.17283818781952) q[13];
cx q[2],q[13];
cx q[3],q[13];
p(9.17283818781952) q[13];
cx q[3],q[13];
cx q[4],q[13];
p(9.17283818781952) q[13];
cx q[4],q[13];
cx q[5],q[13];
p(9.17283818781952) q[13];
cx q[5],q[13];
cx q[6],q[13];
p(9.17283818781952) q[13];
cx q[6],q[13];
cx q[7],q[13];
p(9.17283818781952) q[13];
cx q[7],q[13];
cx q[8],q[13];
p(9.17283818781952) q[13];
cx q[8],q[13];
cx q[9],q[13];
p(9.17283818781952) q[13];
cx q[9],q[13];
cx q[10],q[13];
p(9.17283818781952) q[13];
cx q[10],q[13];
cx q[11],q[13];
p(9.17283818781952) q[13];
cx q[11],q[13];
cx q[12],q[13];
p(9.17283818781952) q[13];
cx q[12],q[13];
u2(2.0,-pi) q[14];
cx q[0],q[14];
p(9.17283818781952) q[14];
cx q[0],q[14];
cx q[1],q[14];
p(9.17283818781952) q[14];
cx q[1],q[14];
cx q[2],q[14];
p(9.17283818781952) q[14];
cx q[2],q[14];
cx q[3],q[14];
p(9.17283818781952) q[14];
cx q[3],q[14];
cx q[4],q[14];
p(9.17283818781952) q[14];
cx q[4],q[14];
cx q[5],q[14];
p(9.17283818781952) q[14];
cx q[5],q[14];
cx q[6],q[14];
p(9.17283818781952) q[14];
cx q[6],q[14];
cx q[7],q[14];
p(9.17283818781952) q[14];
cx q[7],q[14];
cx q[8],q[14];
p(9.17283818781952) q[14];
cx q[8],q[14];
cx q[9],q[14];
p(9.17283818781952) q[14];
cx q[9],q[14];
cx q[10],q[14];
p(9.17283818781952) q[14];
cx q[10],q[14];
cx q[11],q[14];
p(9.17283818781952) q[14];
cx q[11],q[14];
cx q[12],q[14];
p(9.17283818781952) q[14];
cx q[12],q[14];
cx q[13],q[14];
p(9.17283818781952) q[14];
cx q[13],q[14];
u2(2.0,-pi) q[15];
cx q[0],q[15];
p(9.17283818781952) q[15];
cx q[0],q[15];
cx q[1],q[15];
p(9.17283818781952) q[15];
cx q[1],q[15];
cx q[2],q[15];
p(9.17283818781952) q[15];
cx q[2],q[15];
cx q[3],q[15];
p(9.17283818781952) q[15];
cx q[3],q[15];
cx q[4],q[15];
p(9.17283818781952) q[15];
cx q[4],q[15];
cx q[5],q[15];
p(9.17283818781952) q[15];
cx q[5],q[15];
cx q[6],q[15];
p(9.17283818781952) q[15];
cx q[6],q[15];
cx q[7],q[15];
p(9.17283818781952) q[15];
cx q[7],q[15];
cx q[8],q[15];
p(9.17283818781952) q[15];
cx q[8],q[15];
cx q[9],q[15];
p(9.17283818781952) q[15];
cx q[9],q[15];
cx q[10],q[15];
p(9.17283818781952) q[15];
cx q[10],q[15];
cx q[11],q[15];
p(9.17283818781952) q[15];
cx q[11],q[15];
cx q[12],q[15];
p(9.17283818781952) q[15];
cx q[12],q[15];
cx q[13],q[15];
p(9.17283818781952) q[15];
cx q[13],q[15];
cx q[14],q[15];
p(9.17283818781952) q[15];
cx q[14],q[15];
u2(2.0,-pi) q[16];
cx q[0],q[16];
p(9.17283818781952) q[16];
cx q[0],q[16];
cx q[1],q[16];
p(9.17283818781952) q[16];
cx q[1],q[16];
cx q[2],q[16];
p(9.17283818781952) q[16];
cx q[2],q[16];
cx q[3],q[16];
p(9.17283818781952) q[16];
cx q[3],q[16];
cx q[4],q[16];
p(9.17283818781952) q[16];
cx q[4],q[16];
cx q[5],q[16];
p(9.17283818781952) q[16];
cx q[5],q[16];
cx q[6],q[16];
p(9.17283818781952) q[16];
cx q[6],q[16];
cx q[7],q[16];
p(9.17283818781952) q[16];
cx q[7],q[16];
cx q[8],q[16];
p(9.17283818781952) q[16];
cx q[8],q[16];
cx q[9],q[16];
p(9.17283818781952) q[16];
cx q[9],q[16];
cx q[10],q[16];
p(9.17283818781952) q[16];
cx q[10],q[16];
cx q[11],q[16];
p(9.17283818781952) q[16];
cx q[11],q[16];
cx q[12],q[16];
p(9.17283818781952) q[16];
cx q[12],q[16];
cx q[13],q[16];
p(9.17283818781952) q[16];
cx q[13],q[16];
cx q[14],q[16];
p(9.17283818781952) q[16];
cx q[14],q[16];
cx q[15],q[16];
p(9.17283818781952) q[16];
cx q[15],q[16];
u2(2.0,-pi) q[17];
cx q[0],q[17];
p(9.17283818781952) q[17];
cx q[0],q[17];
cx q[1],q[17];
p(9.17283818781952) q[17];
cx q[1],q[17];
cx q[2],q[17];
p(9.17283818781952) q[17];
cx q[2],q[17];
cx q[3],q[17];
p(9.17283818781952) q[17];
cx q[3],q[17];
cx q[4],q[17];
p(9.17283818781952) q[17];
cx q[4],q[17];
cx q[5],q[17];
p(9.17283818781952) q[17];
cx q[5],q[17];
cx q[6],q[17];
p(9.17283818781952) q[17];
cx q[6],q[17];
cx q[7],q[17];
p(9.17283818781952) q[17];
cx q[7],q[17];
cx q[8],q[17];
p(9.17283818781952) q[17];
cx q[8],q[17];
cx q[9],q[17];
p(9.17283818781952) q[17];
cx q[9],q[17];
cx q[10],q[17];
p(9.17283818781952) q[17];
cx q[10],q[17];
cx q[11],q[17];
p(9.17283818781952) q[17];
cx q[11],q[17];
cx q[12],q[17];
p(9.17283818781952) q[17];
cx q[12],q[17];
cx q[13],q[17];
p(9.17283818781952) q[17];
cx q[13],q[17];
cx q[14],q[17];
p(9.17283818781952) q[17];
cx q[14],q[17];
cx q[15],q[17];
p(9.17283818781952) q[17];
cx q[15],q[17];
cx q[16],q[17];
p(9.17283818781952) q[17];
cx q[16],q[17];
u2(2.0,-pi) q[18];
cx q[0],q[18];
p(9.17283818781952) q[18];
cx q[0],q[18];
cx q[1],q[18];
p(9.17283818781952) q[18];
cx q[1],q[18];
cx q[2],q[18];
p(9.17283818781952) q[18];
cx q[2],q[18];
cx q[3],q[18];
p(9.17283818781952) q[18];
cx q[3],q[18];
cx q[4],q[18];
p(9.17283818781952) q[18];
cx q[4],q[18];
cx q[5],q[18];
p(9.17283818781952) q[18];
cx q[5],q[18];
cx q[6],q[18];
p(9.17283818781952) q[18];
cx q[6],q[18];
cx q[7],q[18];
p(9.17283818781952) q[18];
cx q[7],q[18];
cx q[8],q[18];
p(9.17283818781952) q[18];
cx q[8],q[18];
cx q[9],q[18];
p(9.17283818781952) q[18];
cx q[9],q[18];
cx q[10],q[18];
p(9.17283818781952) q[18];
cx q[10],q[18];
cx q[11],q[18];
p(9.17283818781952) q[18];
cx q[11],q[18];
cx q[12],q[18];
p(9.17283818781952) q[18];
cx q[12],q[18];
cx q[13],q[18];
p(9.17283818781952) q[18];
cx q[13],q[18];
cx q[14],q[18];
p(9.17283818781952) q[18];
cx q[14],q[18];
cx q[15],q[18];
p(9.17283818781952) q[18];
cx q[15],q[18];
cx q[16],q[18];
p(9.17283818781952) q[18];
cx q[16],q[18];
cx q[17],q[18];
p(9.17283818781952) q[18];
cx q[17],q[18];
u2(2.0,-pi) q[19];
cx q[0],q[19];
p(9.17283818781952) q[19];
cx q[0],q[19];
cx q[1],q[19];
p(9.17283818781952) q[19];
cx q[1],q[19];
cx q[2],q[19];
p(9.17283818781952) q[19];
cx q[2],q[19];
cx q[3],q[19];
p(9.17283818781952) q[19];
cx q[3],q[19];
cx q[4],q[19];
p(9.17283818781952) q[19];
cx q[4],q[19];
cx q[5],q[19];
p(9.17283818781952) q[19];
cx q[5],q[19];
cx q[6],q[19];
p(9.17283818781952) q[19];
cx q[6],q[19];
cx q[7],q[19];
p(9.17283818781952) q[19];
cx q[7],q[19];
cx q[8],q[19];
p(9.17283818781952) q[19];
cx q[8],q[19];
cx q[9],q[19];
p(9.17283818781952) q[19];
cx q[9],q[19];
cx q[10],q[19];
p(9.17283818781952) q[19];
cx q[10],q[19];
cx q[11],q[19];
p(9.17283818781952) q[19];
cx q[11],q[19];
cx q[12],q[19];
p(9.17283818781952) q[19];
cx q[12],q[19];
cx q[13],q[19];
p(9.17283818781952) q[19];
cx q[13],q[19];
cx q[14],q[19];
p(9.17283818781952) q[19];
cx q[14],q[19];
cx q[15],q[19];
p(9.17283818781952) q[19];
cx q[15],q[19];
cx q[16],q[19];
p(9.17283818781952) q[19];
cx q[16],q[19];
cx q[17],q[19];
p(9.17283818781952) q[19];
cx q[17],q[19];
cx q[18],q[19];
p(9.17283818781952) q[19];
cx q[18],q[19];
u2(2.0,-pi) q[20];
cx q[0],q[20];
p(9.17283818781952) q[20];
cx q[0],q[20];
cx q[1],q[20];
p(9.17283818781952) q[20];
cx q[1],q[20];
cx q[2],q[20];
p(9.17283818781952) q[20];
cx q[2],q[20];
cx q[3],q[20];
p(9.17283818781952) q[20];
cx q[3],q[20];
cx q[4],q[20];
p(9.17283818781952) q[20];
cx q[4],q[20];
cx q[5],q[20];
p(9.17283818781952) q[20];
cx q[5],q[20];
cx q[6],q[20];
p(9.17283818781952) q[20];
cx q[6],q[20];
cx q[7],q[20];
p(9.17283818781952) q[20];
cx q[7],q[20];
cx q[8],q[20];
p(9.17283818781952) q[20];
cx q[8],q[20];
cx q[9],q[20];
p(9.17283818781952) q[20];
cx q[9],q[20];
cx q[10],q[20];
p(9.17283818781952) q[20];
cx q[10],q[20];
cx q[11],q[20];
p(9.17283818781952) q[20];
cx q[11],q[20];
cx q[12],q[20];
p(9.17283818781952) q[20];
cx q[12],q[20];
cx q[13],q[20];
p(9.17283818781952) q[20];
cx q[13],q[20];
cx q[14],q[20];
p(9.17283818781952) q[20];
cx q[14],q[20];
cx q[15],q[20];
p(9.17283818781952) q[20];
cx q[15],q[20];
cx q[16],q[20];
p(9.17283818781952) q[20];
cx q[16],q[20];
cx q[17],q[20];
p(9.17283818781952) q[20];
cx q[17],q[20];
cx q[18],q[20];
p(9.17283818781952) q[20];
cx q[18],q[20];
cx q[19],q[20];
p(9.17283818781952) q[20];
cx q[19],q[20];
u2(2.0,-pi) q[21];
cx q[0],q[21];
p(9.17283818781952) q[21];
cx q[0],q[21];
cx q[1],q[21];
p(9.17283818781952) q[21];
cx q[1],q[21];
cx q[2],q[21];
p(9.17283818781952) q[21];
cx q[2],q[21];
cx q[3],q[21];
p(9.17283818781952) q[21];
cx q[3],q[21];
cx q[4],q[21];
p(9.17283818781952) q[21];
cx q[4],q[21];
cx q[5],q[21];
p(9.17283818781952) q[21];
cx q[5],q[21];
cx q[6],q[21];
p(9.17283818781952) q[21];
cx q[6],q[21];
cx q[7],q[21];
p(9.17283818781952) q[21];
cx q[7],q[21];
cx q[8],q[21];
p(9.17283818781952) q[21];
cx q[8],q[21];
cx q[9],q[21];
p(9.17283818781952) q[21];
cx q[9],q[21];
cx q[10],q[21];
p(9.17283818781952) q[21];
cx q[10],q[21];
cx q[11],q[21];
p(9.17283818781952) q[21];
cx q[11],q[21];
cx q[12],q[21];
p(9.17283818781952) q[21];
cx q[12],q[21];
cx q[13],q[21];
p(9.17283818781952) q[21];
cx q[13],q[21];
cx q[14],q[21];
p(9.17283818781952) q[21];
cx q[14],q[21];
cx q[15],q[21];
p(9.17283818781952) q[21];
cx q[15],q[21];
cx q[16],q[21];
p(9.17283818781952) q[21];
cx q[16],q[21];
cx q[17],q[21];
p(9.17283818781952) q[21];
cx q[17],q[21];
cx q[18],q[21];
p(9.17283818781952) q[21];
cx q[18],q[21];
cx q[19],q[21];
p(9.17283818781952) q[21];
cx q[19],q[21];
cx q[20],q[21];
p(9.17283818781952) q[21];
cx q[20],q[21];
u2(2.0,-pi) q[22];
cx q[0],q[22];
p(9.17283818781952) q[22];
cx q[0],q[22];
cx q[1],q[22];
p(9.17283818781952) q[22];
cx q[1],q[22];
cx q[2],q[22];
p(9.17283818781952) q[22];
cx q[2],q[22];
cx q[3],q[22];
p(9.17283818781952) q[22];
cx q[3],q[22];
cx q[4],q[22];
p(9.17283818781952) q[22];
cx q[4],q[22];
cx q[5],q[22];
p(9.17283818781952) q[22];
cx q[5],q[22];
cx q[6],q[22];
p(9.17283818781952) q[22];
cx q[6],q[22];
cx q[7],q[22];
p(9.17283818781952) q[22];
cx q[7],q[22];
cx q[8],q[22];
p(9.17283818781952) q[22];
cx q[8],q[22];
cx q[9],q[22];
p(9.17283818781952) q[22];
cx q[9],q[22];
cx q[10],q[22];
p(9.17283818781952) q[22];
cx q[10],q[22];
cx q[11],q[22];
p(9.17283818781952) q[22];
cx q[11],q[22];
cx q[12],q[22];
p(9.17283818781952) q[22];
cx q[12],q[22];
cx q[13],q[22];
p(9.17283818781952) q[22];
cx q[13],q[22];
cx q[14],q[22];
p(9.17283818781952) q[22];
cx q[14],q[22];
cx q[15],q[22];
p(9.17283818781952) q[22];
cx q[15],q[22];
cx q[16],q[22];
p(9.17283818781952) q[22];
cx q[16],q[22];
cx q[17],q[22];
p(9.17283818781952) q[22];
cx q[17],q[22];
cx q[18],q[22];
p(9.17283818781952) q[22];
cx q[18],q[22];
cx q[19],q[22];
p(9.17283818781952) q[22];
cx q[19],q[22];
cx q[20],q[22];
p(9.17283818781952) q[22];
cx q[20],q[22];
cx q[21],q[22];
p(9.17283818781952) q[22];
cx q[21],q[22];
u2(2.0,-pi) q[23];
cx q[0],q[23];
p(9.17283818781952) q[23];
cx q[0],q[23];
u2(2.0,-pi) q[0];
cx q[1],q[23];
p(9.17283818781952) q[23];
cx q[1],q[23];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
cx q[2],q[23];
p(9.17283818781952) q[23];
cx q[2],q[23];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
cx q[3],q[23];
p(9.17283818781952) q[23];
cx q[3],q[23];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
cx q[4],q[23];
p(9.17283818781952) q[23];
cx q[4],q[23];
u2(2.0,-pi) q[4];
cx q[0],q[4];
p(9.17283818781952) q[4];
cx q[0],q[4];
cx q[1],q[4];
p(9.17283818781952) q[4];
cx q[1],q[4];
cx q[2],q[4];
p(9.17283818781952) q[4];
cx q[2],q[4];
cx q[3],q[4];
p(9.17283818781952) q[4];
cx q[3],q[4];
cx q[5],q[23];
p(9.17283818781952) q[23];
cx q[5],q[23];
u2(2.0,-pi) q[5];
cx q[0],q[5];
p(9.17283818781952) q[5];
cx q[0],q[5];
cx q[1],q[5];
p(9.17283818781952) q[5];
cx q[1],q[5];
cx q[2],q[5];
p(9.17283818781952) q[5];
cx q[2],q[5];
cx q[3],q[5];
p(9.17283818781952) q[5];
cx q[3],q[5];
cx q[4],q[5];
p(9.17283818781952) q[5];
cx q[4],q[5];
cx q[6],q[23];
p(9.17283818781952) q[23];
cx q[6],q[23];
u2(2.0,-pi) q[6];
cx q[0],q[6];
p(9.17283818781952) q[6];
cx q[0],q[6];
cx q[1],q[6];
p(9.17283818781952) q[6];
cx q[1],q[6];
cx q[2],q[6];
p(9.17283818781952) q[6];
cx q[2],q[6];
cx q[3],q[6];
p(9.17283818781952) q[6];
cx q[3],q[6];
cx q[4],q[6];
p(9.17283818781952) q[6];
cx q[4],q[6];
cx q[5],q[6];
p(9.17283818781952) q[6];
cx q[5],q[6];
cx q[7],q[23];
p(9.17283818781952) q[23];
cx q[7],q[23];
u2(2.0,-pi) q[7];
cx q[0],q[7];
p(9.17283818781952) q[7];
cx q[0],q[7];
cx q[1],q[7];
p(9.17283818781952) q[7];
cx q[1],q[7];
cx q[2],q[7];
p(9.17283818781952) q[7];
cx q[2],q[7];
cx q[3],q[7];
p(9.17283818781952) q[7];
cx q[3],q[7];
cx q[4],q[7];
p(9.17283818781952) q[7];
cx q[4],q[7];
cx q[5],q[7];
p(9.17283818781952) q[7];
cx q[5],q[7];
cx q[6],q[7];
p(9.17283818781952) q[7];
cx q[6],q[7];
cx q[8],q[23];
p(9.17283818781952) q[23];
cx q[8],q[23];
u2(2.0,-pi) q[8];
cx q[0],q[8];
p(9.17283818781952) q[8];
cx q[0],q[8];
cx q[1],q[8];
p(9.17283818781952) q[8];
cx q[1],q[8];
cx q[2],q[8];
p(9.17283818781952) q[8];
cx q[2],q[8];
cx q[3],q[8];
p(9.17283818781952) q[8];
cx q[3],q[8];
cx q[4],q[8];
p(9.17283818781952) q[8];
cx q[4],q[8];
cx q[5],q[8];
p(9.17283818781952) q[8];
cx q[5],q[8];
cx q[6],q[8];
p(9.17283818781952) q[8];
cx q[6],q[8];
cx q[7],q[8];
p(9.17283818781952) q[8];
cx q[7],q[8];
cx q[9],q[23];
p(9.17283818781952) q[23];
cx q[9],q[23];
cx q[10],q[23];
p(9.17283818781952) q[23];
cx q[10],q[23];
u2(2.0,-pi) q[10];
cx q[11],q[23];
p(9.17283818781952) q[23];
cx q[11],q[23];
u2(2.0,-pi) q[11];
cx q[12],q[23];
p(9.17283818781952) q[23];
cx q[12],q[23];
u2(2.0,-pi) q[12];
cx q[13],q[23];
p(9.17283818781952) q[23];
cx q[13],q[23];
u2(2.0,-pi) q[13];
cx q[14],q[23];
p(9.17283818781952) q[23];
cx q[14],q[23];
u2(2.0,-pi) q[14];
cx q[15],q[23];
p(9.17283818781952) q[23];
cx q[15],q[23];
u2(2.0,-pi) q[15];
cx q[16],q[23];
p(9.17283818781952) q[23];
cx q[16],q[23];
u2(2.0,-pi) q[16];
cx q[17],q[23];
p(9.17283818781952) q[23];
cx q[17],q[23];
u2(2.0,-pi) q[17];
cx q[18],q[23];
p(9.17283818781952) q[23];
cx q[18],q[23];
u2(2.0,-pi) q[18];
cx q[19],q[23];
p(9.17283818781952) q[23];
cx q[19],q[23];
u2(2.0,-pi) q[19];
cx q[20],q[23];
p(9.17283818781952) q[23];
cx q[20],q[23];
u2(2.0,-pi) q[20];
cx q[21],q[23];
p(9.17283818781952) q[23];
cx q[21],q[23];
u2(2.0,-pi) q[21];
cx q[22],q[23];
p(9.17283818781952) q[23];
cx q[22],q[23];
u2(2.0,-pi) q[22];
u2(2.0,-pi) q[23];
u2(2.0,-pi) q[9];
cx q[0],q[9];
p(9.17283818781952) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[1],q[9];
p(9.17283818781952) q[10];
cx q[0],q[10];
cx q[0],q[11];
p(9.17283818781952) q[11];
cx q[0],q[11];
cx q[0],q[12];
p(9.17283818781952) q[12];
cx q[0],q[12];
cx q[0],q[13];
p(9.17283818781952) q[13];
cx q[0],q[13];
cx q[0],q[14];
p(9.17283818781952) q[14];
cx q[0],q[14];
cx q[0],q[15];
p(9.17283818781952) q[15];
cx q[0],q[15];
cx q[0],q[16];
p(9.17283818781952) q[16];
cx q[0],q[16];
cx q[0],q[17];
p(9.17283818781952) q[17];
cx q[0],q[17];
cx q[0],q[18];
p(9.17283818781952) q[18];
cx q[0],q[18];
cx q[0],q[19];
p(9.17283818781952) q[19];
cx q[0],q[19];
cx q[0],q[20];
p(9.17283818781952) q[20];
cx q[0],q[20];
cx q[0],q[21];
p(9.17283818781952) q[21];
cx q[0],q[21];
cx q[0],q[22];
p(9.17283818781952) q[22];
cx q[0],q[22];
cx q[0],q[23];
p(9.17283818781952) q[23];
cx q[0],q[23];
ry(0.3629049213995) q[0];
p(9.17283818781952) q[9];
cx q[1],q[9];
cx q[1],q[10];
p(9.17283818781952) q[10];
cx q[1],q[10];
cx q[1],q[11];
p(9.17283818781952) q[11];
cx q[1],q[11];
cx q[1],q[12];
p(9.17283818781952) q[12];
cx q[1],q[12];
cx q[1],q[13];
p(9.17283818781952) q[13];
cx q[1],q[13];
cx q[1],q[14];
p(9.17283818781952) q[14];
cx q[1],q[14];
cx q[1],q[15];
p(9.17283818781952) q[15];
cx q[1],q[15];
cx q[1],q[16];
p(9.17283818781952) q[16];
cx q[1],q[16];
cx q[1],q[17];
p(9.17283818781952) q[17];
cx q[1],q[17];
cx q[1],q[18];
p(9.17283818781952) q[18];
cx q[1],q[18];
cx q[1],q[19];
p(9.17283818781952) q[19];
cx q[1],q[19];
cx q[1],q[20];
cx q[2],q[9];
p(9.17283818781952) q[20];
cx q[1],q[20];
cx q[1],q[21];
p(9.17283818781952) q[21];
cx q[1],q[21];
cx q[1],q[22];
p(9.17283818781952) q[22];
cx q[1],q[22];
cx q[1],q[23];
p(9.17283818781952) q[23];
cx q[1],q[23];
ry(0.011124474286144) q[1];
p(9.17283818781952) q[9];
cx q[2],q[9];
cx q[2],q[10];
p(9.17283818781952) q[10];
cx q[2],q[10];
cx q[2],q[11];
p(9.17283818781952) q[11];
cx q[2],q[11];
cx q[2],q[12];
p(9.17283818781952) q[12];
cx q[2],q[12];
cx q[2],q[13];
p(9.17283818781952) q[13];
cx q[2],q[13];
cx q[2],q[14];
p(9.17283818781952) q[14];
cx q[2],q[14];
cx q[2],q[15];
p(9.17283818781952) q[15];
cx q[2],q[15];
cx q[2],q[16];
p(9.17283818781952) q[16];
cx q[2],q[16];
cx q[2],q[17];
p(9.17283818781952) q[17];
cx q[2],q[17];
cx q[2],q[18];
p(9.17283818781952) q[18];
cx q[2],q[18];
cx q[2],q[19];
p(9.17283818781952) q[19];
cx q[2],q[19];
cx q[2],q[20];
p(9.17283818781952) q[20];
cx q[2],q[20];
cx q[2],q[21];
p(9.17283818781952) q[21];
cx q[2],q[21];
cx q[2],q[22];
p(9.17283818781952) q[22];
cx q[2],q[22];
cx q[2],q[23];
p(9.17283818781952) q[23];
cx q[2],q[23];
ry(0.61901648823278) q[2];
cx q[3],q[9];
p(9.17283818781952) q[9];
cx q[3],q[9];
cx q[3],q[10];
p(9.17283818781952) q[10];
cx q[3],q[10];
cx q[3],q[11];
p(9.17283818781952) q[11];
cx q[3],q[11];
cx q[3],q[12];
p(9.17283818781952) q[12];
cx q[3],q[12];
cx q[3],q[13];
p(9.17283818781952) q[13];
cx q[3],q[13];
cx q[3],q[14];
p(9.17283818781952) q[14];
cx q[3],q[14];
cx q[3],q[15];
p(9.17283818781952) q[15];
cx q[3],q[15];
cx q[3],q[16];
p(9.17283818781952) q[16];
cx q[3],q[16];
cx q[3],q[17];
p(9.17283818781952) q[17];
cx q[3],q[17];
cx q[3],q[18];
p(9.17283818781952) q[18];
cx q[3],q[18];
cx q[3],q[19];
p(9.17283818781952) q[19];
cx q[3],q[19];
cx q[3],q[20];
p(9.17283818781952) q[20];
cx q[3],q[20];
cx q[3],q[21];
p(9.17283818781952) q[21];
cx q[3],q[21];
cx q[3],q[22];
p(9.17283818781952) q[22];
cx q[3],q[22];
cx q[3],q[23];
p(9.17283818781952) q[23];
cx q[3],q[23];
ry(0.548024499535669) q[3];
cx q[4],q[9];
p(9.17283818781952) q[9];
cx q[4],q[9];
cx q[4],q[10];
p(9.17283818781952) q[10];
cx q[4],q[10];
cx q[4],q[11];
p(9.17283818781952) q[11];
cx q[4],q[11];
cx q[4],q[12];
p(9.17283818781952) q[12];
cx q[4],q[12];
cx q[4],q[13];
p(9.17283818781952) q[13];
cx q[4],q[13];
cx q[4],q[14];
p(9.17283818781952) q[14];
cx q[4],q[14];
cx q[4],q[15];
p(9.17283818781952) q[15];
cx q[4],q[15];
cx q[4],q[16];
p(9.17283818781952) q[16];
cx q[4],q[16];
cx q[4],q[17];
p(9.17283818781952) q[17];
cx q[4],q[17];
cx q[4],q[18];
p(9.17283818781952) q[18];
cx q[4],q[18];
cx q[4],q[19];
p(9.17283818781952) q[19];
cx q[4],q[19];
cx q[4],q[20];
p(9.17283818781952) q[20];
cx q[4],q[20];
cx q[4],q[21];
p(9.17283818781952) q[21];
cx q[4],q[21];
cx q[4],q[22];
p(9.17283818781952) q[22];
cx q[4],q[22];
cx q[4],q[23];
p(9.17283818781952) q[23];
cx q[4],q[23];
ry(0.21689358761074) q[4];
cx q[5],q[9];
p(9.17283818781952) q[9];
cx q[5],q[9];
cx q[5],q[10];
p(9.17283818781952) q[10];
cx q[5],q[10];
cx q[5],q[11];
p(9.17283818781952) q[11];
cx q[5],q[11];
cx q[5],q[12];
p(9.17283818781952) q[12];
cx q[5],q[12];
cx q[5],q[13];
p(9.17283818781952) q[13];
cx q[5],q[13];
cx q[5],q[14];
p(9.17283818781952) q[14];
cx q[5],q[14];
cx q[5],q[15];
p(9.17283818781952) q[15];
cx q[5],q[15];
cx q[5],q[16];
p(9.17283818781952) q[16];
cx q[5],q[16];
cx q[5],q[17];
p(9.17283818781952) q[17];
cx q[5],q[17];
cx q[5],q[18];
p(9.17283818781952) q[18];
cx q[5],q[18];
cx q[5],q[19];
p(9.17283818781952) q[19];
cx q[5],q[19];
cx q[5],q[20];
p(9.17283818781952) q[20];
cx q[5],q[20];
cx q[5],q[21];
p(9.17283818781952) q[21];
cx q[5],q[21];
cx q[5],q[22];
p(9.17283818781952) q[22];
cx q[5],q[22];
cx q[5],q[23];
p(9.17283818781952) q[23];
cx q[5],q[23];
ry(0.435495001239459) q[5];
cx q[6],q[9];
p(9.17283818781952) q[9];
cx q[6],q[9];
cx q[6],q[10];
p(9.17283818781952) q[10];
cx q[6],q[10];
cx q[6],q[11];
p(9.17283818781952) q[11];
cx q[6],q[11];
cx q[6],q[12];
p(9.17283818781952) q[12];
cx q[6],q[12];
cx q[6],q[13];
p(9.17283818781952) q[13];
cx q[6],q[13];
cx q[6],q[14];
p(9.17283818781952) q[14];
cx q[6],q[14];
cx q[6],q[15];
p(9.17283818781952) q[15];
cx q[6],q[15];
cx q[6],q[16];
p(9.17283818781952) q[16];
cx q[6],q[16];
cx q[6],q[17];
p(9.17283818781952) q[17];
cx q[6],q[17];
cx q[6],q[18];
p(9.17283818781952) q[18];
cx q[6],q[18];
cx q[6],q[19];
p(9.17283818781952) q[19];
cx q[6],q[19];
cx q[6],q[20];
p(9.17283818781952) q[20];
cx q[6],q[20];
cx q[6],q[21];
p(9.17283818781952) q[21];
cx q[6],q[21];
cx q[6],q[22];
p(9.17283818781952) q[22];
cx q[6],q[22];
cx q[6],q[23];
p(9.17283818781952) q[23];
cx q[6],q[23];
ry(0.80120670218173) q[6];
cx q[7],q[9];
p(9.17283818781952) q[9];
cx q[7],q[9];
cx q[7],q[10];
p(9.17283818781952) q[10];
cx q[7],q[10];
cx q[7],q[11];
p(9.17283818781952) q[11];
cx q[7],q[11];
cx q[7],q[12];
p(9.17283818781952) q[12];
cx q[7],q[12];
cx q[7],q[13];
p(9.17283818781952) q[13];
cx q[7],q[13];
cx q[7],q[14];
p(9.17283818781952) q[14];
cx q[7],q[14];
cx q[7],q[15];
p(9.17283818781952) q[15];
cx q[7],q[15];
cx q[7],q[16];
p(9.17283818781952) q[16];
cx q[7],q[16];
cx q[7],q[17];
p(9.17283818781952) q[17];
cx q[7],q[17];
cx q[7],q[18];
p(9.17283818781952) q[18];
cx q[7],q[18];
cx q[7],q[19];
p(9.17283818781952) q[19];
cx q[7],q[19];
cx q[7],q[20];
p(9.17283818781952) q[20];
cx q[7],q[20];
cx q[7],q[21];
p(9.17283818781952) q[21];
cx q[7],q[21];
cx q[7],q[22];
p(9.17283818781952) q[22];
cx q[7],q[22];
cx q[7],q[23];
p(9.17283818781952) q[23];
cx q[7],q[23];
ry(0.512764785028486) q[7];
cx q[8],q[9];
p(9.17283818781952) q[9];
cx q[8],q[9];
cx q[8],q[10];
p(9.17283818781952) q[10];
cx q[8],q[10];
cx q[8],q[11];
p(9.17283818781952) q[11];
cx q[8],q[11];
cx q[8],q[12];
p(9.17283818781952) q[12];
cx q[8],q[12];
cx q[8],q[13];
p(9.17283818781952) q[13];
cx q[8],q[13];
cx q[8],q[14];
p(9.17283818781952) q[14];
cx q[8],q[14];
cx q[8],q[15];
p(9.17283818781952) q[15];
cx q[8],q[15];
cx q[8],q[16];
p(9.17283818781952) q[16];
cx q[8],q[16];
cx q[8],q[17];
p(9.17283818781952) q[17];
cx q[8],q[17];
cx q[8],q[18];
p(9.17283818781952) q[18];
cx q[8],q[18];
cx q[8],q[19];
p(9.17283818781952) q[19];
cx q[8],q[19];
cx q[8],q[20];
p(9.17283818781952) q[20];
cx q[8],q[20];
cx q[8],q[21];
p(9.17283818781952) q[21];
cx q[8],q[21];
cx q[8],q[22];
p(9.17283818781952) q[22];
cx q[8],q[22];
cx q[8],q[23];
p(9.17283818781952) q[23];
cx q[8],q[23];
ry(0.540814886639395) q[8];
cx q[9],q[10];
p(9.17283818781952) q[10];
cx q[9],q[10];
cx q[9],q[11];
p(9.17283818781952) q[11];
cx q[9],q[11];
cx q[10],q[11];
p(9.17283818781952) q[11];
cx q[10],q[11];
cx q[9],q[12];
p(9.17283818781952) q[12];
cx q[9],q[12];
cx q[10],q[12];
p(9.17283818781952) q[12];
cx q[10],q[12];
cx q[11],q[12];
p(9.17283818781952) q[12];
cx q[11],q[12];
cx q[9],q[13];
p(9.17283818781952) q[13];
cx q[9],q[13];
cx q[10],q[13];
p(9.17283818781952) q[13];
cx q[10],q[13];
cx q[11],q[13];
p(9.17283818781952) q[13];
cx q[11],q[13];
cx q[12],q[13];
p(9.17283818781952) q[13];
cx q[12],q[13];
cx q[9],q[14];
p(9.17283818781952) q[14];
cx q[9],q[14];
cx q[10],q[14];
p(9.17283818781952) q[14];
cx q[10],q[14];
cx q[11],q[14];
p(9.17283818781952) q[14];
cx q[11],q[14];
cx q[12],q[14];
p(9.17283818781952) q[14];
cx q[12],q[14];
cx q[13],q[14];
p(9.17283818781952) q[14];
cx q[13],q[14];
cx q[9],q[15];
p(9.17283818781952) q[15];
cx q[9],q[15];
cx q[10],q[15];
p(9.17283818781952) q[15];
cx q[10],q[15];
cx q[11],q[15];
p(9.17283818781952) q[15];
cx q[11],q[15];
cx q[12],q[15];
p(9.17283818781952) q[15];
cx q[12],q[15];
cx q[13],q[15];
p(9.17283818781952) q[15];
cx q[13],q[15];
cx q[14],q[15];
p(9.17283818781952) q[15];
cx q[14],q[15];
cx q[9],q[16];
p(9.17283818781952) q[16];
cx q[9],q[16];
cx q[10],q[16];
p(9.17283818781952) q[16];
cx q[10],q[16];
cx q[11],q[16];
p(9.17283818781952) q[16];
cx q[11],q[16];
cx q[12],q[16];
p(9.17283818781952) q[16];
cx q[12],q[16];
cx q[13],q[16];
p(9.17283818781952) q[16];
cx q[13],q[16];
cx q[14],q[16];
p(9.17283818781952) q[16];
cx q[14],q[16];
cx q[15],q[16];
p(9.17283818781952) q[16];
cx q[15],q[16];
cx q[9],q[17];
p(9.17283818781952) q[17];
cx q[9],q[17];
cx q[10],q[17];
p(9.17283818781952) q[17];
cx q[10],q[17];
cx q[11],q[17];
p(9.17283818781952) q[17];
cx q[11],q[17];
cx q[12],q[17];
p(9.17283818781952) q[17];
cx q[12],q[17];
cx q[13],q[17];
p(9.17283818781952) q[17];
cx q[13],q[17];
cx q[14],q[17];
p(9.17283818781952) q[17];
cx q[14],q[17];
cx q[15],q[17];
p(9.17283818781952) q[17];
cx q[15],q[17];
cx q[16],q[17];
p(9.17283818781952) q[17];
cx q[16],q[17];
cx q[9],q[18];
p(9.17283818781952) q[18];
cx q[9],q[18];
cx q[10],q[18];
p(9.17283818781952) q[18];
cx q[10],q[18];
cx q[11],q[18];
p(9.17283818781952) q[18];
cx q[11],q[18];
cx q[12],q[18];
p(9.17283818781952) q[18];
cx q[12],q[18];
cx q[13],q[18];
p(9.17283818781952) q[18];
cx q[13],q[18];
cx q[14],q[18];
p(9.17283818781952) q[18];
cx q[14],q[18];
cx q[15],q[18];
p(9.17283818781952) q[18];
cx q[15],q[18];
cx q[16],q[18];
p(9.17283818781952) q[18];
cx q[16],q[18];
cx q[17],q[18];
p(9.17283818781952) q[18];
cx q[17],q[18];
cx q[9],q[19];
p(9.17283818781952) q[19];
cx q[9],q[19];
cx q[10],q[19];
p(9.17283818781952) q[19];
cx q[10],q[19];
cx q[11],q[19];
p(9.17283818781952) q[19];
cx q[11],q[19];
cx q[12],q[19];
p(9.17283818781952) q[19];
cx q[12],q[19];
cx q[13],q[19];
p(9.17283818781952) q[19];
cx q[13],q[19];
cx q[14],q[19];
p(9.17283818781952) q[19];
cx q[14],q[19];
cx q[15],q[19];
p(9.17283818781952) q[19];
cx q[15],q[19];
cx q[16],q[19];
p(9.17283818781952) q[19];
cx q[16],q[19];
cx q[17],q[19];
p(9.17283818781952) q[19];
cx q[17],q[19];
cx q[18],q[19];
p(9.17283818781952) q[19];
cx q[18],q[19];
cx q[9],q[20];
p(9.17283818781952) q[20];
cx q[9],q[20];
cx q[10],q[20];
p(9.17283818781952) q[20];
cx q[10],q[20];
cx q[11],q[20];
p(9.17283818781952) q[20];
cx q[11],q[20];
cx q[12],q[20];
p(9.17283818781952) q[20];
cx q[12],q[20];
cx q[13],q[20];
p(9.17283818781952) q[20];
cx q[13],q[20];
cx q[14],q[20];
p(9.17283818781952) q[20];
cx q[14],q[20];
cx q[15],q[20];
p(9.17283818781952) q[20];
cx q[15],q[20];
cx q[16],q[20];
p(9.17283818781952) q[20];
cx q[16],q[20];
cx q[17],q[20];
p(9.17283818781952) q[20];
cx q[17],q[20];
cx q[18],q[20];
p(9.17283818781952) q[20];
cx q[18],q[20];
cx q[19],q[20];
p(9.17283818781952) q[20];
cx q[19],q[20];
cx q[9],q[21];
p(9.17283818781952) q[21];
cx q[9],q[21];
cx q[10],q[21];
p(9.17283818781952) q[21];
cx q[10],q[21];
cx q[11],q[21];
p(9.17283818781952) q[21];
cx q[11],q[21];
cx q[12],q[21];
p(9.17283818781952) q[21];
cx q[12],q[21];
cx q[13],q[21];
p(9.17283818781952) q[21];
cx q[13],q[21];
cx q[14],q[21];
p(9.17283818781952) q[21];
cx q[14],q[21];
cx q[15],q[21];
p(9.17283818781952) q[21];
cx q[15],q[21];
cx q[16],q[21];
p(9.17283818781952) q[21];
cx q[16],q[21];
cx q[17],q[21];
p(9.17283818781952) q[21];
cx q[17],q[21];
cx q[18],q[21];
p(9.17283818781952) q[21];
cx q[18],q[21];
cx q[19],q[21];
p(9.17283818781952) q[21];
cx q[19],q[21];
cx q[20],q[21];
p(9.17283818781952) q[21];
cx q[20],q[21];
cx q[9],q[22];
p(9.17283818781952) q[22];
cx q[9],q[22];
cx q[10],q[22];
p(9.17283818781952) q[22];
cx q[10],q[22];
cx q[11],q[22];
p(9.17283818781952) q[22];
cx q[11],q[22];
cx q[12],q[22];
p(9.17283818781952) q[22];
cx q[12],q[22];
cx q[13],q[22];
p(9.17283818781952) q[22];
cx q[13],q[22];
cx q[14],q[22];
p(9.17283818781952) q[22];
cx q[14],q[22];
cx q[15],q[22];
p(9.17283818781952) q[22];
cx q[15],q[22];
cx q[16],q[22];
p(9.17283818781952) q[22];
cx q[16],q[22];
cx q[17],q[22];
p(9.17283818781952) q[22];
cx q[17],q[22];
cx q[18],q[22];
p(9.17283818781952) q[22];
cx q[18],q[22];
cx q[19],q[22];
p(9.17283818781952) q[22];
cx q[19],q[22];
cx q[20],q[22];
p(9.17283818781952) q[22];
cx q[20],q[22];
cx q[21],q[22];
p(9.17283818781952) q[22];
cx q[21],q[22];
cx q[9],q[23];
p(9.17283818781952) q[23];
cx q[9],q[23];
cx q[10],q[23];
p(9.17283818781952) q[23];
cx q[10],q[23];
ry(0.558466394083638) q[10];
cx q[11],q[23];
p(9.17283818781952) q[23];
cx q[11],q[23];
ry(0.961411658008286) q[11];
cx q[12],q[23];
p(9.17283818781952) q[23];
cx q[12],q[23];
ry(0.114537444994351) q[12];
cx q[13],q[23];
p(9.17283818781952) q[23];
cx q[13],q[23];
ry(0.376007530625489) q[13];
cx q[14],q[23];
p(9.17283818781952) q[23];
cx q[14],q[23];
ry(0.242177768969065) q[14];
cx q[15],q[23];
p(9.17283818781952) q[23];
cx q[15],q[23];
ry(0.370404115884164) q[15];
cx q[16],q[23];
p(9.17283818781952) q[23];
cx q[16],q[23];
ry(0.360460849180617) q[16];
cx q[17],q[23];
p(9.17283818781952) q[23];
cx q[17],q[23];
ry(0.446430682783108) q[17];
cx q[18],q[23];
p(9.17283818781952) q[23];
cx q[18],q[23];
ry(0.547687554768277) q[18];
cx q[19],q[23];
p(9.17283818781952) q[23];
cx q[19],q[23];
ry(0.255406777182917) q[19];
cx q[20],q[23];
p(9.17283818781952) q[23];
cx q[20],q[23];
ry(0.0558831763784048) q[20];
cx q[21],q[23];
p(9.17283818781952) q[23];
cx q[21],q[23];
ry(0.578531895917143) q[21];
cx q[22],q[23];
p(9.17283818781952) q[23];
cx q[22],q[23];
ry(0.942948907459552) q[22];
ry(0.809464629257493) q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[20],q[21];
cx q[19],q[20];
cx q[18],q[19];
cx q[17],q[18];
cx q[16],q[17];
cx q[15],q[16];
cx q[14],q[15];
cx q[13],q[14];
cx q[12],q[13];
cx q[11],q[12];
cx q[10],q[11];
ry(0.690735237555462) q[11];
ry(0.338222341827594) q[12];
ry(0.117361262339738) q[13];
ry(0.731779970119676) q[14];
ry(0.254365703216971) q[15];
ry(0.87509972004931) q[16];
ry(0.312120697533516) q[17];
ry(0.671464367848413) q[18];
ry(0.412502656852618) q[19];
ry(0.439439779880891) q[20];
ry(0.0740412784944793) q[21];
ry(0.635422506141274) q[22];
ry(0.447553973024752) q[23];
ry(0.58588457981835) q[9];
cx q[9],q[10];
ry(0.907974944381626) q[10];
cx q[8],q[9];
cx q[7],q[8];
cx q[6],q[7];
cx q[5],q[6];
cx q[4],q[5];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(0.876296434655476) q[0];
ry(0.359941937385196) q[1];
ry(0.505747681445153) q[2];
ry(0.380051181105455) q[3];
ry(0.257662407132846) q[4];
ry(0.157398618828393) q[5];
ry(0.71546083895452) q[6];
ry(0.799291309560875) q[7];
ry(0.121066306460545) q[8];
ry(0.261763531255604) q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13],q[14],q[15],q[16],q[17],q[18],q[19],q[20],q[21],q[22],q[23];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];
meas[9] = measure q[9];
meas[10] = measure q[10];
meas[11] = measure q[11];
meas[12] = measure q[12];
meas[13] = measure q[13];
meas[14] = measure q[14];
meas[15] = measure q[15];
meas[16] = measure q[16];
meas[17] = measure q[17];
meas[18] = measure q[18];
meas[19] = measure q[19];
meas[20] = measure q[20];
meas[21] = measure q[21];
meas[22] = measure q[22];
meas[23] = measure q[23];