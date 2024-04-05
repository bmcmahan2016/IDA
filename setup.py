from setuptools import setup, find_packages

setup(
    name='IDA',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'your_script_name = your_package_name.module_name:main_function',
        ],
    },
    # Metadata
    author='Brandon McMahan',
    description='Interventional Diffusion Assistance (IDA) provides a way for a copilot  \
    to assist a human user in various tasks',
    url='https://your.package.url',
)
