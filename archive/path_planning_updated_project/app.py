#!/usr/bin/env python3
"""
Evasive Maneuvers Visualizer - Backend API
"""
import json
import math
import os
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np

from evasive_maneuvers import (
    PlatformType,
    ManeuverType,
    EvasiveTargetGenerator,
    ProportionalNavigationInterceptor,
    compute_engagement_kinematics,
    G_ACCEL,
)

# Cube space parameters
CUBE_SIZE = 10000.0  # meters
BODY_DIAG = math.sqrt(3) * CUBE_SIZE

# Global simulation state
current_simulation = None

class EvasiveAPIHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type='application/json'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/maneuvers':
            self._set_headers()
            maneuvers = []
            for m in ManeuverType:
                maneuvers.append({
                    'id': m.value,
                    'name': m.name
                })
            self.wfile.write(json.dumps(maneuvers).encode())
        
        elif path == '/api/params':
            self._set_headers()
            params = {
                'cube_size': CUBE_SIZE,
                'body_diag': BODY_DIAG,
                'time_factor': 1.75
            }
            self.wfile.write(json.dumps(params).encode())
        
        elif path == '/' or path == '/index.html':
            self._serve_file('index.html')
        elif path.endswith('.js') or path.endswith('.css') or path.endswith('.html'):
            self._serve_file(path.lstrip('/'))
        else:
            self._set_headers()
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/generate_trajectory':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            traj_type = data.get('type', 'polynomial')  # polynomial or bezier
            order = data.get('order', 2)
            point_count = data.get('point_count', 4)
            eta = data.get('eta', 0.5)
            
            trajectory = self._generate_trajectory(traj_type, order, point_count, eta)
            
            self._set_headers()
            self.wfile.write(json.dumps(trajectory).encode())
        
        elif path == '/api/simulate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            result = self._run_simulation(data)
            global current_simulation
            current_simulation = result
            
            self._set_headers()
            self.wfile.write(json.dumps(result).encode())
        
        else:
            self._set_headers()
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def _serve_file(self, filename):
        try:
            filepath = os.path.join(os.path.dirname(__file__), filename)
            with open(filepath, 'rb') as f:
                content = f.read()
            if filename.endswith('.html'):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
            elif filename.endswith('.js'):
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
            elif filename.endswith('.css'):
                self.send_response(200)
                self.send_header('Content-type', 'text/css')
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/octet-stream')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f'File not found: {e}'.encode())
    
    def _generate_trajectory(self, traj_type, order, point_count, eta):
        """Generate trajectory points"""
        points = []
        if traj_type == 'polynomial':
            # Wall to wall in cube
            # Start at wall, end at wall
            # Generate random polynomial coefficients
            start_x = random.choice([0, CUBE_SIZE])
            start_y = random.uniform(0, CUBE_SIZE)
            start_z = random.uniform(0, CUBE_SIZE)
            end_x = CUBE_SIZE - start_x if start_x == 0 else 0
            end_y = random.uniform(0, CUBE_SIZE)
            end_z = random.uniform(0, CUBE_SIZE)
            
            # Generate control points for polynomial
            t_vals = np.linspace(0, 1, 100)
            for t in t_vals:
                # Simple polynomial interpolation
                x = start_x * (1-t)**order + end_x * t**order + random.uniform(-100, 100)
                y = start_y * (1-t)**order + end_y * t**order
                z = start_z * (1-t)**order + end_z * t**order
                # Clamp to cube
                x = max(0, min(CUBE_SIZE, x))
                y = max(0, min(CUBE_SIZE, y))
                z = max(0, min(CUBE_SIZE, z))
                points.append([x, y, z])
        else:  # bezier
            # Generate n points, spread apart
            min_dist = eta * BODY_DIAG / 2.0
            ctrl_points = []
            # Start point
            ctrl_points.append([random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE)])
            for i in range(1, point_count):
                # Try to place points far apart
                attempts = 0
                while attempts < 100:
                    p = [random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE)]
                    # Check distance from previous
                    if np.linalg.norm(np.array(p) - np.array(ctrl_points[-1])) >= min_dist:
                        ctrl_points.append(p)
                        break
                    attempts += 1
                if attempts >= 100:
                    ctrl_points.append([random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE), random.uniform(0, CUBE_SIZE)])
            
            # Evaluate bezier curve
            n = len(ctrl_points) - 1
            t_vals = np.linspace(0, 1, 100)
            for t in t_vals:
                # Bernstein polynomials
                pt = [0, 0, 0]
                for i in range(n + 1):
                    coeff = math.comb(n, i) * (t**i) * ((1-t)**(n-i))
                    pt[0] += coeff * ctrl_points[i][0]
                    pt[1] += coeff * ctrl_points[i][1]
                    pt[2] += coeff * ctrl_points[i][2]
                points.append(pt)
        
        return {
            'points': points,
            'type': traj_type,
            'order': order if traj_type == 'polynomial' else point_count
        }
    
    def _run_simulation(self, data):
        """Run simulation"""
        try:
            # Extract parameters
            traj_points = data.get('trajectory', [])
            interceptor_pos_2d = data.get('interceptor_pos', [5000, 5000])  # [x, y] on ground
            maneuver_id = data.get('maneuver', 'none')
            max_vel = data.get('max_vel', 200)
            max_accel = data.get('max_accel', 30)
            
            if not traj_points or len(traj_points) == 0:
                return {'error': 'No trajectory'}
            
            # Calculate target speed for constant time
            max_drone_speed = data.get('max_drone_speed', 150)
            total_time = 1.75 * BODY_DIAG / max_drone_speed if max_drone_speed > 0 else 15.0
            
            dt = 0.02
            steps = max(10, int(total_time / dt))
            
            # Initial conditions - give target initial velocity along trajectory
            p_T = np.array(traj_points[0], dtype=float)
            if len(traj_points) > 1:
                dir_T = np.array(traj_points[1]) - np.array(traj_points[0])
                dir_norm = np.linalg.norm(dir_T)
                if dir_norm > 1e-6:
                    dir_T = dir_T / dir_norm
                else:
                    dir_T = np.array([1, 0, 0])
            else:
                dir_T = np.array([1, 0, 0])
            v_T = dir_T * max(60.0, max_drone_speed * 0.4)
            a_T = np.array([0, 0, 0])
            
            # Interceptor at 2D position on ground, pointed toward target
            p_M = np.array([float(interceptor_pos_2d[0]), float(interceptor_pos_2d[1]), 50.0])
            # Initial velocity toward target
            r0 = p_T - p_M
            r0_norm = np.linalg.norm(r0)
            if r0_norm > 1e-6:
                v_M_dir = r0 / r0_norm
            else:
                v_M_dir = np.array([1, 0, 0])
            v_M = v_M_dir * max(120.0, float(max_vel) * 0.6)
            
            # Get maneuver type
            try:
                maneuver = ManeuverType(maneuver_id)
            except Exception:
                maneuver = ManeuverType.NONE
            
            # Use fixed wing for better intercept
            target_platform = data.get('target_platform', 'fixed_wing')
            try:
                plat_t = PlatformType(target_platform)
            except Exception:
                plat_t = PlatformType.FIXED_WING
            
            target = EvasiveTargetGenerator(plat_t, p_T, v_T, maneuver)
            nav_ratio = float(data.get('nav_ratio', 3.0))
            a_m_max = float(max_accel) * G_ACCEL
            interceptor = ProportionalNavigationInterceptor(p_M, v_M, N_nav=nav_ratio, tau_M=0.25, a_M_max=a_m_max, use_apn=True)
            
            traj_T = []
            traj_M = []
            miss_dist = float('inf')
            a_M_new = np.array([0,0,0])
            v_M_new = np.array([0,0,0])
            
            for i in range(steps):
                if maneuver == ManeuverType.NONE and len(traj_points) > 0:
                    t_path = min(1.0, (i * dt) / total_time)
                    idx = int(t_path * (len(traj_points) - 1))
                    p_T_new = np.array(traj_points[idx], dtype=float)
                    if idx < len(traj_points) - 1:
                        v_T_new = (np.array(traj_points[idx + 1]) - np.array(traj_points[idx])) / dt
                    else:
                        v_T_new = np.array(v_T)
                    a_T_new = np.array([0.0, 0.0, 0.0])
                    kin = compute_engagement_kinematics(p_T_new, v_T_new, interceptor.p_W, interceptor.v_W)
                else:
                    p_T_new, v_T_new, a_T_new, kin = target.step(dt, i*dt, interceptor.p_W, interceptor.v_W)
                p_M_new, v_M_new_res, a_M_new_res = interceptor.step(dt, p_T_new, v_T_new, a_T_new)
                v_M_new = v_M_new_res
                a_M_new = a_M_new_res
                
                traj_T.append([float(x) for x in p_T_new])
                traj_M.append([float(x) for x in p_M_new])
                
                dist = float(np.linalg.norm(p_T_new - p_M_new))
                if dist < miss_dist:
                    miss_dist = dist
                
                p_T, v_T, a_T = p_T_new, v_T_new, a_T_new
                if dist < 2.0:
                    break
                if hasattr(kin, 'v_cl') and kin.v_cl < -5.0 and dist > miss_dist + 10.0:
                    break
            
            if len(traj_T) == 0 and len(traj_points) > 0:
                traj_T = [[float(x) for x in traj_points[0]]]
            elif len(traj_T) == 0:
                traj_T = [[float(x) for x in p_T]]
            if len(traj_M) == 0:
                traj_M = [[float(x) for x in p_M]]
            
            return {
                'traj_T': traj_T,
                'traj_M': traj_M,
                'miss_distance': float(miss_dist),
                'total_time': float(total_time),
                'final_a_T': [float(x) for x in (a_T if hasattr(a_T, '__iter__') else [0,0,0])],
                'final_v_T': [float(x) for x in (v_T if hasattr(v_T, '__iter__') else [0,0,0])],
                'final_a_M': [float(x) for x in a_M_new],
                'final_v_M': [float(x) for x in v_M_new],
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, EvasiveAPIHandler)
    print(f'Server running on http://localhost:{port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
