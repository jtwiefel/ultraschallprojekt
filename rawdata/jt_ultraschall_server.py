# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:02:25 2018

@author: twiefel
"""




from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import numpy as np
from cPickle import dumps
import os

class UltraschallServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/python-pickle')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        
        os.system("sudo insmod ./scope.ko")
        os.system("sudo mknod /dev/chardev c 243 0")

        device_name = "/dev/chardev"
        
        #for dummy acquisition
        #device_name =  "hannes28.dat"
        #print device_name

        raw_data = np.fromfile(device_name, dtype = '<i4')
        self.wfile.write(dumps(raw_data))
        
        os.system("sudo rm /dev/chardev")
        os.system("sudo rmmod scope.ko")

        
def run(server_class=HTTPServer, handler_class=UltraschallServer, port=8000):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()   
    
if __name__ == "__main__":
    run()