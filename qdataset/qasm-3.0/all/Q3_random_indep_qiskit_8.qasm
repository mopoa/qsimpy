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
gate cu1 (lambda) a, b{  
  
  u3(0,0,lambda/2) a;
  cx a,b;
  u3(0,0,-lambda/2) b;
  cx a,b;
  u3(0,0,lambda/2) b; 
  
}

// controlled-U
gate cu3(theta,phi,lambda) c, t
{
  // implements controlled-U(theta,phi,lambda) with  target t and control c
  u3(0,0,(lambda+phi)/2) c;
  u3(0,0,(lambda-phi)/2) t;
  cx c,t;
  u3(-theta/2,0,-(phi+lambda)/2) t;
  cx c,t;
  u3(theta/2,phi,0) t;
}

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
u2(0.4263905139762012,2.6735518805676657) q[0];
tdg q[1];
cu3(4.684618627355123,0.8755706304852969,5.695888160944528) q[0],q[1];
u2(0.9388015080041642,0.8594715742269348) q[0];
rx(pi/2) q[1];
u2(0,0) q[3];
sdg q[4];
u2(pi/4,-pi) q[5];
u1(-0.38304509451314317) q[6];
cx q[3],q[6];
ry(-0.4270040294696573) q[3];
ry(-0.4270040294696573) q[6];
cx q[3],q[6];
u1(0.3830450945131423) q[6];
cx q[7],q[5];
tdg q[5];
cx q[2],q[5];
u1(-0.35385558524315597) q[2];
cx q[3],q[2];
ry(-2.680531730054245) q[2];
ry(2.680531730054245) q[3];
cx q[3],q[2];
u1(0.35385558524315597) q[2];
u2(-pi,-pi) q[3];
t q[5];
cx q[7],q[5];
u1(-pi/4) q[5];
ccx q[4],q[6],q[5];
h q[4];
cx q[3],q[4];
h q[4];
cu1(pi/2) q[3],q[4];
u3(1.2063647973960818,2.1549166954047863,1.8578290617371813) q[4];
u3(1.4955045655857289,-1.67912483173346,0) q[5];
cu3(1.5693858735638762,5.465706745619813,3.772222268266399) q[6],q[2];
crx(6.202284721500185) q[2],q[0];
tdg q[0];
u2(0,-pi/2) q[2];
u2(1.255084050953391,-2.3007450194397885) q[6];
cx q[5],q[6];
ry(2.7703663541572343) q[5];
ry(-2.7703663541572343) q[6];
cx q[5],q[6];
u2(-pi,-pi) q[5];
cy q[0],q[5];
u1(0.3048374864941841) q[6];
cu1(pi/2) q[6],q[5];
u1(-2.2593201473701177) q[5];
s q[6];
u2(pi/2,-pi/2) q[7];
rzz(1.5640163777242566) q[1],q[7];
u3(3.0668278331846897,-2.183186840075576,-1.2124915129657658) q[1];
rx(-pi/2) q[7];
ch q[7],q[3];
cu1(pi/2) q[3],q[4];
u2(4.29663826721623,3.2899582493108444) q[3];
u2(-pi/2,pi/2) q[4];
s q[7];
cx q[2],q[7];
cx q[7],q[2];
u3(1.1684206606722325,3.072779991711874,-2.490120072622904) q[2];
u2(0,-pi/2) q[7];
rzz(2.917738905811407) q[7],q[4];
rx(-pi/2) q[4];
rzz(0.6044551176832745) q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
s q[2];
u3(1.0312348538591947,-pi,-pi/2) q[3];
t q[4];
rx(-pi/2) q[7];
cz q[7],q[0];
sx q[0];
cx q[6],q[0];
u2(0,0) q[0];
u2(-pi,pi/2) q[6];
cx q[6],q[2];
cx q[2],q[6];
h q[2];

// rxx gate
h q[2];
h q[6];
cx q[2],q[6];
rz(2.867615720945729) q[6];
cx q[2],q[6];
h q[6];
h q[2];

h q[2];
cx q[3],q[2];
h q[2];
cu1(pi/2) q[3],q[2];
u2(0,-pi/2) q[2];
u1(-pi) q[3];
u3(1.620475581832197,0.04685466475022748,1.4510178277509702) q[6];
cx q[6],q[3];
cx q[3],q[6];
u3(1.4638877630605924,-2.87168576673058,0.12287974816374536) q[3];
u2(-pi,-pi) q[6];
u2(0,0) q[7];
cx q[7],q[5];
ry(-0.06243932267886539) q[5];
ry(-0.06243932267886539) q[7];
cx q[7],q[5];
u1(2.2593201473701185) q[5];
cu1(pi/2) q[4],q[5];
h q[4];
cu1(pi/2) q[5],q[4];
u2(-pi,pi/2) q[4];
cx q[0],q[4];
x q[0];
h q[4];
u3(0.7817431109358302,0.15527430675967135,1.5185179918312786) q[5];
u2(-pi,0.398674155836213) q[7];
cx q[7],q[1];
h q[7];
cx q[1],q[7];
rz(5.442492121247214) q[7];
cx q[1],q[7];
swap q[1],q[0];
rx(pi/2) q[0];
ry(3.62694954519184) q[1];
rzz(0.8536767566826966) q[2],q[0];
rx(-pi/2) q[0];
cu1(pi/8) q[0],q[6];
cx q[0],q[5];
u3(pi,0,-pi/2) q[2];
cu1(-pi/8) q[5],q[6];
cx q[0],q[5];
cu1(pi/8) q[5],q[6];
cx q[5],q[1];
cu1(-pi/8) q[1],q[6];
cx q[0],q[1];
cu1(pi/8) q[1],q[6];
cx q[5],q[1];
cu1(-pi/8) q[1],q[6];
cx q[0],q[1];
u1(2.683738114930269) q[0];
cu1(pi/8) q[1],q[6];
u1(2.2958382109065414) q[1];
p(4.213907225096717) q[5];
h q[7];
cx q[7],q[4];
h q[4];
cu1(pi/2) q[7],q[4];
u2(-pi/2,pi/2) q[4];
rx(pi/2) q[7];
rzz(5.376266498080888) q[4],q[7];
u3(3*pi/4,-pi/2,-pi) q[4];
cu1(pi/2) q[3],q[4];
h q[4];
u3(0.9877279055030962,2.5134344796136956,2.6655844175327497) q[7];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];