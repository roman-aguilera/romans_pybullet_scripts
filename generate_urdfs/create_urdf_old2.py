import inspect #used to list attributes of objecte
import os

home_dir = os.getenv("HOME")


path_to_xml_file = os.path.join(home_dir, "Documents/RoboLab/bullet3/examples/pybullet/gym/pybullet_data/romans_urdf_files/octopus_files/base_plane.urdf")

import xml.etree.ElementTree as ET
tree = ET.parse(path_to_xml_file)
root = tree.getroot()

print(root, root.tag, root.attrib)
print(inspect.getmembers(root))
print()
print(list(root))

print("arguments take by subelement")
print(ET.SubElement)

#args_for_subelement = {'parent':'root', 'tag':'link','attrib':'{'name': 'link_1'}'}

number_of_links = 90 #3
# information on link properties 
# http://wiki.ros.org/urdf/XML/link
for link_index in range(1, number_of_links+1):
	link = ET.SubElement( root, 'link', {'name': 'link_' + str(link_index)}  )
	
	collision = ET.SubElement(link, 'collision' ) #collision = ET.SubElement(link, 'collision' , {'': '' } )
	
	geometry = ET.SubElement( collision, 'geometry' ) #geometry = ET.SubElement( collision, 'geometry', {'': '' } )
	
	cylinder = ET.SubElement( geometry, 'cylinder', {'length':'0.6', 'radius':'0.2'} )
	
	origin = ET.SubElement( collision, 'origin', {'xyz':'0 0 1'} )

# information on joint properties 
# http://wiki.ros.org/urdf/XML/joint#Attributes	
for joint_index in range(1, number_of_links + 1):
	joint = ET.SubElement( root, 'joint' , {'name':'link_'+ str(joint_index-1) + '_to_link_'+ str(joint_index) , 'type':'continuous'} ) 
	#joint = ET.SubElement( root, 'joint'+ str(joint_index-1) + 'link'+ str(joint_index) , {'name':'link_'+ str(joint_index-1) + '_to_link_'+ str(joint_index) , 'type':'continuous'} )
	
	parent = ET.SubElement( joint , 'parent', {'link':'link_'+str(joint_index-1)} )
	child = ET.SubElement( joint , 'child', {'link':'link_'+str(joint_index)} )
	origin = ET.SubElement( joint , 'origin', {'xyz':'0 0 2'} )
	axis = ET.SubElement( joint , 'xyz', {'xyz':'1 0 0'} )
	
#tree structure is as follows
# #robot(root)
	##link
		###collision
			####geometry
				#####cylinder
			####geometry
			####origin
		###collision
	##link
	##joint
		###parent
		###child
		###origin
	##joint
# #robot	

#ET.SubElement( args_for_subelement )



"""
for i in root:
        for j in i:
                for k in j:
                        for l in k:
                                print(l, k, i, j)
"""
name_of_generated_xml_file = "octopus_generated_"+ str(number_of_links) + "_links.xml"
name_of_generated_urdf_file = "octopus_generated_"+ str(number_of_links) + "_links.urdf"

tree.write( name_of_generated_xml_file )
tree.write( name_of_generated_urdf_file )


