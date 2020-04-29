from setuptools import setup


setup(name='hwsim',
      packages=[
          'hwsim'
      ],
      version = '0.0.1',
      description='Python wrapper for the hwsim library',
      url='https://github.com/dcbr',
      author='Bram De Cooman',
      install_requires=['numpy','h5py','pyvista','vtk','imageio-ffmpeg'],
      include_package_data=True,
      package_data = {
          'hwsim': ['*.dll','*.so','*.obj'],
          }
)