######################################################################
#
# flags

MY_CXX_FLAGS:=$('-Wno-unused-variable -Wno-all')
MY_LIBS:=""
MY_LDFLAGS:=""

######################################################################
#
# Template libMesh application Makefile
LIBMESH_DIR = /home/chaztikov/src/libmesh/libmesh-1.3.1/clang8


# include the library options determined by configure
include $(LIBMESH_DIR)/Make.common

target     := ./example-$(METHOD)


###############################################################################
# File management.  This is where the source, header, and object files are
# defined

#
# source files
srcfiles 	:= $(wildcard *.C)

#
# object files
objects		:= $(patsubst %.C, %.$(obj-suffix), $(srcfiles))
###############################################################################



.PHONY: dust clean distclean

###############################################################################
# Target:
#

all:: $(notdir $(target))

# Production rules:  how to make the target - depends on library configuration
$(notdir $(target)): $(objects)
	@echo "Linking "$@"..."
	@$(libmesh_LIBTOOL) --tag=CXX $(LIBTOOLFLAGS) --mode=link \
	  $(libmesh_CXX) $(libmesh_CXXFLAGS) $(MY_CXX_FLAGS) $(objects) -o $@ $(libmesh_LIBS) $(MY_LIBS) $(libmesh_LDFLAGS) $(MY_LDFLAGS) $(EXTERNAL_FLAGS)


# Useful rules.
dust:
	@echo "Deleting old output and runtime files"
	@rm -f out*.m job_output.txt output.txt* *.gmv.* *.plt.* out*.xdr* out*.xda* PI* complete

clean: dust
	@rm -f $(objects) *.$(obj-suffix) *.lo

clobber: clean 
	@rm -f $(target)

distclean: clean
	@rm -rf *.o .libs

echo:
	@echo srcfiles = $(srcfiles)
	@echo objects = $(objects)
	@echo target = $(target)

run: complete

complete: $(wildcard *.in)
#	@$(MAKE) dust
	@$(MAKE) -C $(dir $(target)) $(notdir $(target))
	@echo "***************************************************************"
	@echo "* Running App " $(notdir $(target))
	@echo "***************************************************************"
	@echo " "
	${LIBMESH_RUN} $(target) ${LIBMESH_OPTIONS} 2>&1 | tee output.txt
	@bzip2 -f output.txt
	@echo " "
	@echo "***************************************************************"
	@echo "* Done Running App " $(notdir $(target))
	@echo "***************************************************************"

###############################################################################
