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

        device_name = "/dev/chardev"

        #for dummy acquisition

        #device_name =  "hannes28.dat"
        #import random
        #device_name = "hannes"+str(random.randint(28,30))+".dat"
        
        try:
            if device_name == "/dev/chardev":
                print "inserting kernel module"
                print os.popen("sudo insmod ./scope.ko").read()
                print "making device node"
                print os.popen("sudo mknod /dev/chardev c 243 0").read()
                #os.system("sudo insmod ./scope.ko")
                #os.system("sudo mknod /dev/chardev c 243 0")
                print "cating to tmp file"
                print os.popen("cat /dev/chardev > tmp.dat")#
            print "opening tmp tmp file"
            raw_data = np.fromfile("tmp.dat", dtype = '<i4')
            print "writing to socket"
            #raw_data = np.fromfile(device_name, dtype = '<i4')
            self.wfile.write(dumps(raw_data))
            print "finished writing"
            
        finally:
            if device_name == "/dev/chardev":
                print "removing tmp file"
                print os.popen("rm tmp.dat")
                print "removing device node"
                print os.popen("sudo rm /dev/chardev").read()
                print "removing kernel module"
                print os.popen("sudo rmmod scope.ko").read()
                print "closing"
                
                #os.system("sudo rm /dev/chardev")
                #os.system("sudo rmmod scope.ko")

        
def run(server_class=HTTPServer, handler_class=UltraschallServer, port=8000):
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()   
    
if __name__ == "__main__":
    run()