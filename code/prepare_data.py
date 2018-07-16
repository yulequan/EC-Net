import collections
import json
import multiprocessing
import os
import sys
import time
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
from subprocess import Popen
import h5py
import numpy as np
from tqdm import tqdm
from GKNN import GKNN

NUM_EDGE = 120
NUM_FACE = 800


class Mesh:
    def __init__(self):
        self.name = None
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.graph = None

    def writeToObjFile(self, pathToObjFile):
        objFile = open(pathToObjFile, 'w')
        objFile.write("# off2obj OBJ File")
        objFile.write("# http://johnsresearch.wordpress.com\n")
        for vert in self.verts:
            objFile.write("v ")
            objFile.write(str(vert[0]))
            objFile.write(" ")
            objFile.write(str(vert[1]))
            objFile.write(" ")
            objFile.write(str(vert[2]))
            objFile.write("\n")
        objFile.write("s off\n")
        for face in self.faces:
            objFile.write("f ")
            objFile.write(str(face[0]+1))
            objFile.write(" ")
            objFile.write(str(face[1]+1))
            objFile.write(" ")
            objFile.write(str(face[2]+1))
            objFile.write("\n")
        objFile.close()

    def loadFromOffFile(self, pathToOffFile,is_remove_reducent=True,is_normalized=True):
        #Reset this mesh:
        self.verts = []
        self.faces = []
        self.nVerts = 0
        self.nFaces = 0
        self.graph = None
        self.name = pathToOffFile.split('/')[-1][:-4]

        #Open the file for reading:
        offFile = open(pathToOffFile, 'r')
        lines = offFile.readlines()

        #Read the number of verts and faces
        if lines[0]!='OFF\n' and lines[0]!='OFF\r\n':
            params=lines[0][3:].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])
            vertLines = lines[1:1 + self.nVerts]
            faceLines = lines[1 + self.nVerts:1 + self.nVerts + self.nFaces]
        else:
            params = lines[1].split()
            self.nVerts = int(params[0])
            self.nFaces = int(params[1])
            vertLines = lines[2:2+self.nVerts]
            faceLines = lines[2+self.nVerts:2+self.nVerts+self.nFaces]
        if is_remove_reducent:
            diffvertLines = []
            index = {}
            for id, vertLine in enumerate(vertLines):
                if vertLine not in diffvertLines:
                    diffvertLines.append(vertLine)
                    XYZ = vertLine.split()
                    self.verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
                index[id] = diffvertLines.index(vertLine)
            for faceLine in faceLines:
                XYZ = faceLine.split()
                self.faces.append([index[int(XYZ[1])], index[int(XYZ[2])], index[int(XYZ[3])]])
                if not (int(XYZ[0]) == 3):
                    print "ERROR: This OFF loader can only handle meshes with 3 vertex faces."
                    print "A face with", XYZ[0], "vertices is included in the file. Exiting."
                    sys.exit(0)
            self.nVerts = len(self.verts)
            self.nFaces = len(self.faces)
        else:
            for vertLine in vertLines:
                XYZ = vertLine.split()
                self.verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
            for faceLine in faceLines:
                XYZ = faceLine.split()
                self.faces.append((int(XYZ[1]), int(XYZ[2]), int(XYZ[3])))
                if not (int(XYZ[0]) == 3):
                    print "ERROR: This OFF loader can only handle meshes with 3 vertex faces."
                    print "A face with", XYZ[0], "vertices is included in the file. Exiting."
                    sys.exit(0)
            self.nVerts = len(self.verts)
            self.nFaces = len(self.faces)

        # unique_id = np.unique(np.asarray(self.faces))
        # self.verts = np.asarray(self.verts)
        # self.verts = self.verts[unique_id]
        # for item in xrange(self.nFaces):
        #     aa = self.faces[item]
        #     bb = [np.where(unique_id==aa[0])[0][0],np.where(unique_id==aa[1])[0][0],np.where(unique_id==aa[2])[0][0]]
        #     self.faces[item] = bb
        # self.nVerts = len(self.verts)
        # self.nFaces = len(self.faces)

        if is_normalized:
            #normalize vertices
            self.verts = np.asarray(self.verts)
            self.centroid = np.mean(self.verts,axis=0,keepdims=True)
            self.verts = self.verts-self.centroid
            self.furthest_dist = np.amax(np.sqrt(np.sum(self.verts*self.verts,axis=1)))
            self.verts = self.verts/self.furthest_dist


    def buildGraph(self):
        if not(self.graph == None):
            return self.graph
        self.graph = []
        for i in range(0, self.nVerts):
            self.graph.append(set())
        for face in self.faces:
            i = face[0]
            j = face[1]
            k = face[2]
            if not(j in self.graph[i]):
                self.graph[i].add(j)
            if not(k in self.graph[i]):
                self.graph[i].add(k)
            if not(i in self.graph[j]):
                self.graph[j].add(i)
            if not(k in self.graph[j]):
                self.graph[j].add(k)
            if not(i in self.graph[k]):
                self.graph[k].add(i)
            if not(j in self.graph[k]):
                self.graph[k].add(j)
        return self.graph



    def write2OffFile(self,path):
        with open(path,'w') as f:
            f.write('OFF\n')
            f.write('%d %d 0\n'%(self.nVerts,self.nFaces))
            for item in self.verts:
                f.write("%0.6f %0.6f %0.6f\n"%(item[0],item[1],item[2]))
            for item in self.faces:
                f.write("3 %d %d %d\n"%(item[0],item[1],item[2]))


    def crop_patchs(self,num_patches,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        idx = np.random.permutation(self.nVerts)[:num_patches]
        patch_size = int(0.5*self.nVerts)

        for i,item in enumerate(idx):
            visited = self.bfs_connected_component(item,patch_size)
            print i, len(visited)
            self.write4visited(visited,save_path+self.name+"_"+str(i)+".off")
        return


    def bfs_connected_component(self, start,num):
        assert(start<self.nVerts)
        q = collections.deque()
        visited = set()
        q.append(start)
        while q:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                if len(visited)>num:
                    break
                q.extend(self.graph[vertex] - visited)
        return list(visited)

    def write4visited(self,visited,save_name):
        index = {}
        for i,item in enumerate(visited):
            index[item] = i
        verts = [self.verts[item] for item in visited]
        faces =[]
        for item in self.faces:
            if item[0] in visited and item[1] in visited and item[2] in visited:
                faces.append([index[item[0]],index[item[1]],index[item[2]]])
        verts = np.asarray(verts)
        centroid = np.mean(verts,axis=0,keepdims=True)
        verts = verts-centroid
        furthest_dist = np.amax(np.sqrt(np.sum(verts*verts,axis=1)))
        verts = verts/furthest_dist

        with open(save_name,'w') as f:
            f.write('OFF\n')
            f.write('%d %d 0\n'%(len(verts),len(faces)))
            for item in verts:
                f.write("%0.6f %0.6f %0.6f\n"%(item[0],item[1],item[2]))
            for item in faces:
                f.write("3 %d %d %d\n"%(item[0],item[1],item[2]))

    def remove_redundent(self, path, save_path=None):
        if save_path==None:
            save_path = path
        self.loadFromOffFile(path)
        self.write2OffFile(save_path+'/'+self.name+".off")

    def preprocess_mesh(self,path,save_path):
        mark = self.loadFromOffFile(path)
        if mark==False:
            return
        self.buildGraph()
        vertexs= set(range(self.nVerts))

        submeshes = []
        while len(vertexs)>0:
            q = collections.deque()
            visited = set()
            q.append(vertexs.pop())
            while q:
                vertex = q.popleft()
                if vertex not in visited:
                    visited.add(vertex)
                    vertexs.discard(vertex)
                    q.extend(self.graph[vertex] - visited)
            if len(visited) < 10:
                continue
            submeshes.append(visited)

        if len(submeshes)>30:
            shuffle(submeshes)
            submeshes = submeshes[:30]
        for i,item in enumerate(submeshes):
            self.write4visited(item,save_path+'/'+self.name+"_"+str(i)+".off")
        print "Total %d submeshes"%(len(submeshes))


def read_annotation_boxes(path):
    name = path.split()[-1][:-4]
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    verts = []
    for item in lines:
        if item[0] == 'v':
            XYZ = item.split()[1:]
            verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
    verts = np.asarray(verts)

    boxes = []
    points = []
    for line_iter in xrange(len(lines)):
        if 'g Box' in lines[line_iter]:
            box = []
            for i in xrange(6):
                vertex = lines[line_iter + i + 1].split()[1:]
                vertex = [int(item) - 1 for item in vertex]
                box.append([verts[vertex[0]], verts[vertex[1]], verts[vertex[2]], verts[vertex[3]]])
            boxes.append(box)
    boxes = np.asarray(boxes)

    boxes = np.asarray(boxes)  # (n,6,4,3)
    return boxes

def read_face(path):
    with open(path,'r') as f:
        lines = f.readlines()
    nVerts = int(lines[1].split()[0])
    nFaces = int(lines[1].split()[1])
    verts = []
    faces = []
    for item in lines[2:2+nVerts]:
        XYZ = item.split()
        verts.append([float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])
    for item in lines[2+nVerts:2+nVerts+nFaces]:
        XYZ = item.split()
        faces.append([verts[int(XYZ[1])], verts[int(XYZ[2])], verts[int(XYZ[3])]])
    faces = np.asarray(faces)
    faces = np.reshape(faces,[-1,9])
    return faces


def save_h5():
    file_list = glob('./patch4k/noise_half/*.xyz')
    file_list.sort()

    mc8k_inputs = []
    mc8k_dists = []
    edge_points = []
    edges = []
    faces = []
    names = []
    for item in tqdm(file_list):
        name = item.split('/')[-1]
        try:
            data = np.loadtxt(item.replace('/noise_half','/noise_half_dist'))
            edge = np.loadtxt(item.replace('/noise_half','/noise_half_edge') )
            edge_point = np.loadtxt(item.replace('/noise_half','/noise_half_edgepoint'))
            face = np.loadtxt(item.replace('/noise_half','/noise_half_face'))
        except:
            print name
            continue
        if edge.shape[0]==0 or data.shape[0]==0:
            print "empty", name
            continue
        if len(edge.shape) == 1:
            edge = np.reshape(edge,[1,-1])

        mc8k_inputs.append(data[:, 0:3])
        mc8k_dists.append(data[:, 3])

        face = np.reshape(face,[-1,9])
        l = face.shape[0]
        idx = range(l) * (NUM_FACE / l) + range(l)[:NUM_FACE % l]
        assert face[idx].shape[0]==NUM_FACE
        assert face[idx].shape[1]==9
        faces.append(face[idx])

        l = len(edge_point)
        idx = range(l) * (2000 / l) + list(np.random.permutation(l)[:2000 % l])
        edge_points.append(edge_point[idx])

        idx = np.all(edge[:, 0:3] == edge[:, 3:6], axis=-1)
        edge = edge[idx==False]
        l = edge.shape[0]
        idx = range(l)*(NUM_EDGE/l)+ range(l)[:NUM_EDGE%l]
        edges.append(edge[idx])
        names.append(name)

    faces = np.asarray(faces)
    print len(names)

    h5_filename = '../h5data/training_data_%d_%d.h5'%(NUM_EDGE,NUM_FACE)
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('mc8k_input', data=mc8k_inputs, compression='gzip', compression_opts=4,dtype=np.float32)
    h5_fout.create_dataset('mc8k_dist', data=mc8k_dists, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge', data=edges, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('edge_points', data=edge_points, compression='gzip', compression_opts=4, dtype=np.float32)
    h5_fout.create_dataset('face', data=faces, compression='gzip', compression_opts=4, dtype=np.float32)
    string_dt = h5py.special_dtype(vlen=str)
    h5_fout.create_dataset('name', data=names, compression='gzip', compression_opts=1, dtype=string_dt)
    h5_fout.close()


ids = [0,2,4,5,6,6,5,4,2,0]
def crop_patch_from_wholepointcloud(off_path):
    current = multiprocessing.current_process()
    id = int(current.name.split('-')[-1])

    print off_path
    point_path = './mesh_simu_pc/' + off_path.split('/')[-1][:-4] + '_noise_half.xyz'
    edge_path = './mesh_edge/' + off_path.split('/')[-1][:-4] + '_edge.xyz'

    if not os.path.exists(edge_path):
        return
    save_root_path = './patch4k/noise_half'
    if 'chair' in off_path or 'bookshelf' in off_path:
        patch_num = 100
    else:
        patch_num = 50

    gm = GKNN(point_path, edge_path, off_path, patch_size=1024, patch_num=patch_num)
    gm.crop_patch(save_root_path, id=ids[id-1], scale_ratio=2.0)


def handle_patch(filter_path=False):
    new_file_list = glob('./mesh/*.off')
    new_file_list.sort()
    pool = multiprocessing.Pool(10)
    pool.map(crop_patch_from_wholepointcloud, new_file_list)



if __name__ == '__main__':
    np.random.seed(int(time.time()))
    os.chdir('../traindata')
    # preprocessing_data()
    # handle_patch()
    save_h5()

