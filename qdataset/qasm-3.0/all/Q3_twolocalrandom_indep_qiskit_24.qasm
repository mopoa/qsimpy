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
ry(4.84635053289793) q[0];
ry(0.130388343310326) q[1];
cx q[0],q[1];
ry(3.98132927960905) q[2];
cx q[0],q[2];
cx q[1],q[2];
ry(4.70487355272564) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
ry(3.13221193522563) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
ry(1.41243898030268) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
ry(1.24446568175557) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
ry(4.77855539654732) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
ry(1.06255472357457) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
ry(0.555055422457116) q[9];
cx q[0],q[9];
cx q[1],q[9];
cx q[2],q[9];
cx q[3],q[9];
cx q[4],q[9];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
ry(4.30624274089981) q[10];
cx q[0],q[10];
cx q[1],q[10];
cx q[2],q[10];
cx q[3],q[10];
cx q[4],q[10];
cx q[5],q[10];
cx q[6],q[10];
cx q[7],q[10];
cx q[8],q[10];
cx q[9],q[10];
ry(5.99034706477481) q[11];
cx q[0],q[11];
cx q[1],q[11];
cx q[2],q[11];
cx q[3],q[11];
cx q[4],q[11];
cx q[5],q[11];
cx q[6],q[11];
cx q[7],q[11];
cx q[8],q[11];
cx q[9],q[11];
cx q[10],q[11];
ry(0.024807688980384) q[12];
cx q[0],q[12];
cx q[1],q[12];
cx q[2],q[12];
cx q[3],q[12];
cx q[4],q[12];
cx q[5],q[12];
cx q[6],q[12];
cx q[7],q[12];
cx q[8],q[12];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
ry(3.21819890375657) q[13];
cx q[0],q[13];
cx q[1],q[13];
cx q[2],q[13];
cx q[3],q[13];
cx q[4],q[13];
cx q[5],q[13];
cx q[6],q[13];
cx q[7],q[13];
cx q[8],q[13];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
ry(5.10584808655871) q[14];
cx q[0],q[14];
cx q[1],q[14];
cx q[2],q[14];
cx q[3],q[14];
cx q[4],q[14];
cx q[5],q[14];
cx q[6],q[14];
cx q[7],q[14];
cx q[8],q[14];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
ry(3.84861478336691) q[15];
cx q[0],q[15];
cx q[1],q[15];
cx q[2],q[15];
cx q[3],q[15];
cx q[4],q[15];
cx q[5],q[15];
cx q[6],q[15];
cx q[7],q[15];
cx q[8],q[15];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
ry(4.53492240586622) q[16];
cx q[0],q[16];
cx q[1],q[16];
cx q[2],q[16];
cx q[3],q[16];
cx q[4],q[16];
cx q[5],q[16];
cx q[6],q[16];
cx q[7],q[16];
cx q[8],q[16];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
ry(1.83391142304707) q[17];
cx q[0],q[17];
cx q[1],q[17];
cx q[2],q[17];
cx q[3],q[17];
cx q[4],q[17];
cx q[5],q[17];
cx q[6],q[17];
cx q[7],q[17];
cx q[8],q[17];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
ry(5.76654488188296) q[18];
cx q[0],q[18];
cx q[1],q[18];
cx q[2],q[18];
cx q[3],q[18];
cx q[4],q[18];
cx q[5],q[18];
cx q[6],q[18];
cx q[7],q[18];
cx q[8],q[18];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
ry(4.48981206311071) q[19];
cx q[0],q[19];
cx q[1],q[19];
cx q[2],q[19];
cx q[3],q[19];
cx q[4],q[19];
cx q[5],q[19];
cx q[6],q[19];
cx q[7],q[19];
cx q[8],q[19];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
ry(3.40890680158139) q[20];
cx q[0],q[20];
cx q[1],q[20];
cx q[2],q[20];
cx q[3],q[20];
cx q[4],q[20];
cx q[5],q[20];
cx q[6],q[20];
cx q[7],q[20];
cx q[8],q[20];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
ry(0.893280754210937) q[21];
cx q[0],q[21];
cx q[1],q[21];
cx q[2],q[21];
cx q[3],q[21];
cx q[4],q[21];
cx q[5],q[21];
cx q[6],q[21];
cx q[7],q[21];
cx q[8],q[21];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
ry(2.34576917812665) q[22];
cx q[0],q[22];
cx q[1],q[22];
cx q[2],q[22];
cx q[3],q[22];
cx q[4],q[22];
cx q[5],q[22];
cx q[6],q[22];
cx q[7],q[22];
cx q[8],q[22];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
ry(4.23570642526072) q[23];
cx q[0],q[23];
ry(2.77611970975908) q[0];
cx q[1],q[23];
ry(2.72699034602209) q[1];
cx q[0],q[1];
cx q[2],q[23];
ry(3.88154440237914) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[23];
ry(3.2241426661697) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[23];
ry(4.0865660173428) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[23];
ry(3.77643912107403) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[23];
ry(5.05936655933969) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[23];
ry(3.27760572345175) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[23];
ry(5.7092092972822) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[23];
cx q[10],q[23];
ry(0.568372854235992) q[10];
cx q[11],q[23];
ry(1.88935417772466) q[11];
cx q[12],q[23];
ry(0.716184867709297) q[12];
cx q[13],q[23];
ry(5.20675833379049) q[13];
cx q[14],q[23];
ry(0.294658264947335) q[14];
cx q[15],q[23];
ry(3.93507820834554) q[15];
cx q[16],q[23];
ry(3.44058528928674) q[16];
cx q[17],q[23];
ry(5.14773201355748) q[17];
cx q[18],q[23];
ry(1.25002425820944) q[18];
cx q[19],q[23];
ry(5.38374923085481) q[19];
cx q[20],q[23];
ry(2.20949869731062) q[20];
cx q[21],q[23];
ry(4.74159128751739) q[21];
cx q[22],q[23];
ry(1.85958224815419) q[22];
ry(5.55393670085889) q[23];
ry(2.0058195038543) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
cx q[0],q[12];
cx q[0],q[13];
cx q[0],q[14];
cx q[0],q[15];
cx q[0],q[16];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[19];
cx q[0],q[20];
cx q[0],q[21];
cx q[0],q[22];
cx q[0],q[23];
ry(2.04524994014355) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
cx q[1],q[12];
cx q[1],q[13];
cx q[1],q[14];
cx q[1],q[15];
cx q[1],q[16];
cx q[1],q[17];
cx q[1],q[18];
cx q[1],q[19];
cx q[1],q[20];
cx q[1],q[21];
cx q[1],q[22];
cx q[1],q[23];
ry(1.036825464) q[1];
cx q[0],q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
cx q[2],q[12];
cx q[2],q[13];
cx q[2],q[14];
cx q[2],q[15];
cx q[2],q[16];
cx q[2],q[17];
cx q[2],q[18];
cx q[2],q[19];
cx q[2],q[20];
cx q[2],q[21];
cx q[2],q[22];
cx q[2],q[23];
ry(2.46633397820351) q[2];
cx q[0],q[2];
cx q[1],q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
cx q[3],q[12];
cx q[3],q[13];
cx q[3],q[14];
cx q[3],q[15];
cx q[3],q[16];
cx q[3],q[17];
cx q[3],q[18];
cx q[3],q[19];
cx q[3],q[20];
cx q[3],q[21];
cx q[3],q[22];
cx q[3],q[23];
ry(0.587228852230412) q[3];
cx q[0],q[3];
cx q[1],q[3];
cx q[2],q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
cx q[4],q[12];
cx q[4],q[13];
cx q[4],q[14];
cx q[4],q[15];
cx q[4],q[16];
cx q[4],q[17];
cx q[4],q[18];
cx q[4],q[19];
cx q[4],q[20];
cx q[4],q[21];
cx q[4],q[22];
cx q[4],q[23];
ry(5.15915900496302) q[4];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
cx q[5],q[12];
cx q[5],q[13];
cx q[5],q[14];
cx q[5],q[15];
cx q[5],q[16];
cx q[5],q[17];
cx q[5],q[18];
cx q[5],q[19];
cx q[5],q[20];
cx q[5],q[21];
cx q[5],q[22];
cx q[5],q[23];
ry(0.949716148968678) q[5];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
cx q[4],q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
cx q[6],q[12];
cx q[6],q[13];
cx q[6],q[14];
cx q[6],q[15];
cx q[6],q[16];
cx q[6],q[17];
cx q[6],q[18];
cx q[6],q[19];
cx q[6],q[20];
cx q[6],q[21];
cx q[6],q[22];
cx q[6],q[23];
ry(2.41346226029822) q[6];
cx q[0],q[6];
cx q[1],q[6];
cx q[2],q[6];
cx q[3],q[6];
cx q[4],q[6];
cx q[5],q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
cx q[7],q[12];
cx q[7],q[13];
cx q[7],q[14];
cx q[7],q[15];
cx q[7],q[16];
cx q[7],q[17];
cx q[7],q[18];
cx q[7],q[19];
cx q[7],q[20];
cx q[7],q[21];
cx q[7],q[22];
cx q[7],q[23];
ry(5.93296503328577) q[7];
cx q[0],q[7];
cx q[1],q[7];
cx q[2],q[7];
cx q[3],q[7];
cx q[4],q[7];
cx q[5],q[7];
cx q[6],q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
cx q[8],q[12];
cx q[8],q[13];
cx q[8],q[14];
cx q[8],q[15];
cx q[8],q[16];
cx q[8],q[17];
cx q[8],q[18];
cx q[8],q[19];
cx q[8],q[20];
cx q[8],q[21];
cx q[8],q[22];
cx q[8],q[23];
ry(6.2054338728997) q[8];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
cx q[4],q[8];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
cx q[9],q[23];
cx q[10],q[23];
ry(5.19068291435812) q[10];
cx q[11],q[23];
ry(1.57943026665478) q[11];
cx q[12],q[23];
ry(3.75339676308994) q[12];
cx q[13],q[23];
ry(5.67265925137076) q[13];
cx q[14],q[23];
ry(3.35872664974762) q[14];
cx q[15],q[23];
ry(3.70834453218737) q[15];
cx q[16],q[23];
ry(0.246814622670602) q[16];
cx q[17],q[23];
ry(2.24423917784508) q[17];
cx q[18],q[23];
ry(0.500223798327118) q[18];
cx q[19],q[23];
ry(1.91926127086387) q[19];
cx q[20],q[23];
ry(2.07797072184667) q[20];
cx q[21],q[23];
ry(4.86211914740084) q[21];
cx q[22],q[23];
ry(0.251071112927388) q[22];
ry(2.69857894507022) q[23];
ry(2.86704602590518) q[9];
cx q[0],q[9];
cx q[0],q[10];
cx q[0],q[11];
cx q[0],q[12];
cx q[0],q[13];
cx q[0],q[14];
cx q[0],q[15];
cx q[0],q[16];
cx q[0],q[17];
cx q[0],q[18];
cx q[0],q[19];
cx q[0],q[20];
cx q[0],q[21];
cx q[0],q[22];
cx q[0],q[23];
ry(1.97874389399801) q[0];
cx q[1],q[9];
cx q[1],q[10];
cx q[1],q[11];
cx q[1],q[12];
cx q[1],q[13];
cx q[1],q[14];
cx q[1],q[15];
cx q[1],q[16];
cx q[1],q[17];
cx q[1],q[18];
cx q[1],q[19];
cx q[1],q[20];
cx q[1],q[21];
cx q[1],q[22];
cx q[1],q[23];
ry(3.99919179827194) q[1];
cx q[2],q[9];
cx q[2],q[10];
cx q[2],q[11];
cx q[2],q[12];
cx q[2],q[13];
cx q[2],q[14];
cx q[2],q[15];
cx q[2],q[16];
cx q[2],q[17];
cx q[2],q[18];
cx q[2],q[19];
cx q[2],q[20];
cx q[2],q[21];
cx q[2],q[22];
cx q[2],q[23];
ry(2.17616332456639) q[2];
cx q[3],q[9];
cx q[3],q[10];
cx q[3],q[11];
cx q[3],q[12];
cx q[3],q[13];
cx q[3],q[14];
cx q[3],q[15];
cx q[3],q[16];
cx q[3],q[17];
cx q[3],q[18];
cx q[3],q[19];
cx q[3],q[20];
cx q[3],q[21];
cx q[3],q[22];
cx q[3],q[23];
ry(0.270788675285506) q[3];
cx q[4],q[9];
cx q[4],q[10];
cx q[4],q[11];
cx q[4],q[12];
cx q[4],q[13];
cx q[4],q[14];
cx q[4],q[15];
cx q[4],q[16];
cx q[4],q[17];
cx q[4],q[18];
cx q[4],q[19];
cx q[4],q[20];
cx q[4],q[21];
cx q[4],q[22];
cx q[4],q[23];
ry(5.52867009609533) q[4];
cx q[5],q[9];
cx q[5],q[10];
cx q[5],q[11];
cx q[5],q[12];
cx q[5],q[13];
cx q[5],q[14];
cx q[5],q[15];
cx q[5],q[16];
cx q[5],q[17];
cx q[5],q[18];
cx q[5],q[19];
cx q[5],q[20];
cx q[5],q[21];
cx q[5],q[22];
cx q[5],q[23];
ry(4.7955820429843) q[5];
cx q[6],q[9];
cx q[6],q[10];
cx q[6],q[11];
cx q[6],q[12];
cx q[6],q[13];
cx q[6],q[14];
cx q[6],q[15];
cx q[6],q[16];
cx q[6],q[17];
cx q[6],q[18];
cx q[6],q[19];
cx q[6],q[20];
cx q[6],q[21];
cx q[6],q[22];
cx q[6],q[23];
ry(5.51724392385255) q[6];
cx q[7],q[9];
cx q[7],q[10];
cx q[7],q[11];
cx q[7],q[12];
cx q[7],q[13];
cx q[7],q[14];
cx q[7],q[15];
cx q[7],q[16];
cx q[7],q[17];
cx q[7],q[18];
cx q[7],q[19];
cx q[7],q[20];
cx q[7],q[21];
cx q[7],q[22];
cx q[7],q[23];
ry(2.62328731818401) q[7];
cx q[8],q[9];
cx q[8],q[10];
cx q[8],q[11];
cx q[8],q[12];
cx q[8],q[13];
cx q[8],q[14];
cx q[8],q[15];
cx q[8],q[16];
cx q[8],q[17];
cx q[8],q[18];
cx q[8],q[19];
cx q[8],q[20];
cx q[8],q[21];
cx q[8],q[22];
cx q[8],q[23];
ry(3.80495605495645) q[8];
cx q[9],q[10];
cx q[9],q[11];
cx q[10],q[11];
cx q[9],q[12];
cx q[10],q[12];
cx q[11],q[12];
cx q[9],q[13];
cx q[10],q[13];
cx q[11],q[13];
cx q[12],q[13];
cx q[9],q[14];
cx q[10],q[14];
cx q[11],q[14];
cx q[12],q[14];
cx q[13],q[14];
cx q[9],q[15];
cx q[10],q[15];
cx q[11],q[15];
cx q[12],q[15];
cx q[13],q[15];
cx q[14],q[15];
cx q[9],q[16];
cx q[10],q[16];
cx q[11],q[16];
cx q[12],q[16];
cx q[13],q[16];
cx q[14],q[16];
cx q[15],q[16];
cx q[9],q[17];
cx q[10],q[17];
cx q[11],q[17];
cx q[12],q[17];
cx q[13],q[17];
cx q[14],q[17];
cx q[15],q[17];
cx q[16],q[17];
cx q[9],q[18];
cx q[10],q[18];
cx q[11],q[18];
cx q[12],q[18];
cx q[13],q[18];
cx q[14],q[18];
cx q[15],q[18];
cx q[16],q[18];
cx q[17],q[18];
cx q[9],q[19];
cx q[10],q[19];
cx q[11],q[19];
cx q[12],q[19];
cx q[13],q[19];
cx q[14],q[19];
cx q[15],q[19];
cx q[16],q[19];
cx q[17],q[19];
cx q[18],q[19];
cx q[9],q[20];
cx q[10],q[20];
cx q[11],q[20];
cx q[12],q[20];
cx q[13],q[20];
cx q[14],q[20];
cx q[15],q[20];
cx q[16],q[20];
cx q[17],q[20];
cx q[18],q[20];
cx q[19],q[20];
cx q[9],q[21];
cx q[10],q[21];
cx q[11],q[21];
cx q[12],q[21];
cx q[13],q[21];
cx q[14],q[21];
cx q[15],q[21];
cx q[16],q[21];
cx q[17],q[21];
cx q[18],q[21];
cx q[19],q[21];
cx q[20],q[21];
cx q[9],q[22];
cx q[10],q[22];
cx q[11],q[22];
cx q[12],q[22];
cx q[13],q[22];
cx q[14],q[22];
cx q[15],q[22];
cx q[16],q[22];
cx q[17],q[22];
cx q[18],q[22];
cx q[19],q[22];
cx q[20],q[22];
cx q[21],q[22];
cx q[9],q[23];
cx q[10],q[23];
ry(3.75631844257445) q[10];
cx q[11],q[23];
ry(1.64754958933665) q[11];
cx q[12],q[23];
ry(1.89043018768814) q[12];
cx q[13],q[23];
ry(0.15959153738279) q[13];
cx q[14],q[23];
ry(1.9041982282388) q[14];
cx q[15],q[23];
ry(1.52100758355808) q[15];
cx q[16],q[23];
ry(3.50336708260893) q[16];
cx q[17],q[23];
ry(3.55318539846825) q[17];
cx q[18],q[23];
ry(2.98534395592546) q[18];
cx q[19],q[23];
ry(1.83970394259416) q[19];
cx q[20],q[23];
ry(0.403701320528425) q[20];
cx q[21],q[23];
ry(6.15010207501049) q[21];
cx q[22],q[23];
ry(2.13444733187909) q[22];
ry(3.11048228390001) q[23];
ry(3.22620596905881) q[9];
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