package umich.ML.mcwrap.map;

import net.minecraft.util.BlockPos;
import umich.ML.mcwrap.MCWrap;
import net.minecraft.entity.EntityLiving;
import net.minecraft.block.state.IBlockState;

public class BlockEntity {
	public int id; 
	public IBlockState block;
	public EntityLiving entity;
	
	public BlockEntity(int id, EntityLiving ent) { 
		this.id = id; 
		this.entity = ent; 
		this.block = null;
	}
	
	public BlockEntity(int id, IBlockState state) { 
		this.id = id; 
		this.entity = null; 
		this.block = state;
	}
	
	public Boolean isBlock() {
		return this.entity == null && this.block != null;
	}
	
	public Boolean isEntity() {
		return this.entity != null && this.block == null;
	}
	
	public void removeFromWorld(BlockPos pos) {
		if (isBlock()) {
			MCWrap.world.setBlockToAir(pos);
		} else {
			assert(isEntity());
			// this.entity.setInvisible(true);
			// MCWrap.world.removeEntity(this.entity);
			
			BlockPos inv_pos = new BlockPos(-1, -1, -1);
			while(Math.abs(this.entity.lastTickPosX - (inv_pos.getX() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.lastTickPosY - inv_pos.getY()) > 0.01 ||
	                Math.abs(this.entity.lastTickPosZ - (inv_pos.getZ() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.posX - (inv_pos.getX() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.posY - inv_pos.getY()) > 0.01 ||
	                Math.abs(this.entity.posZ - (inv_pos.getZ() + 0.5)) > 0.01) {
				this.entity.setPositionAndUpdate(inv_pos.getX() + 0.5, inv_pos.getY(), inv_pos.getZ() + 0.5);
				this.entity.onEntityUpdate();
				// this.entity.onLivingUpdate();
				this.entity.onUpdate();
			}

			this.entity.setDead();
			this.entity.worldObj.removeEntity(this.entity);
		}
	}
	
	public void addEntityToWorld(BlockPos pos) {
		if (isBlock()) {
			MCWrap.world.setBlockState(pos, this.block);
		} else {
			assert(isEntity());
			MCWrap.world.spawnEntityInWorld(this.entity);
			this.entity.setEntityBoundingBox(null);
			this.entity.motionX = 0;
			this.entity.motionY = 0;
			this.entity.motionZ = 0;
			
			while(Math.abs(this.entity.lastTickPosX - (pos.getX() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.lastTickPosY - pos.getY()) > 0.01 ||
	                Math.abs(this.entity.lastTickPosZ - (pos.getZ() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.posX - (pos.getX() + 0.5)) > 0.01 ||
	                Math.abs(this.entity.posY - pos.getY()) > 0.01 ||
	                Math.abs(this.entity.posZ - (pos.getZ() + 0.5)) > 0.01) {
				this.entity.setPositionAndUpdate(pos.getX() + 0.5, pos.getY(), pos.getZ() + 0.5);
				this.entity.onEntityUpdate();
				// this.entity.onLivingUpdate();
				this.entity.onUpdate();
			}
		}
	}
	
	public void update() {
		if (isEntity()) {
			int iter = 0;
			while (true) {
				double prev_x = this.entity.getLookVec().xCoord;
				double prev_y = this.entity.getLookVec().yCoord;
				double prev_z = this.entity.getLookVec().zCoord;
				double prev_yaw = this.entity.prevRotationYaw;
				double prev_pitch = this.entity.prevRotationPitch;
				
				this.entity.getLookHelper().setLookPosition(MCWrap.player.posX, 
						MCWrap.player.posY + MCWrap.player.getEyeHeight(), 
						MCWrap.player.posZ, 10.0F, 
						(float)this.entity.getVerticalFaceSpeed());
				this.entity.getLookHelper().onUpdateLook();
				this.entity.onEntityUpdate();
				this.entity.onUpdate();
				
				double x = this.entity.getLookVec().xCoord;
				double y = this.entity.getLookVec().yCoord;
				double z = this.entity.getLookVec().zCoord;
				double yaw = this.entity.rotationYaw;
				double pitch = this.entity.rotationPitch;
				
				iter = iter + 1;
				if (Math.abs(x - prev_x) < 0.01 && 
					Math.abs(y - prev_y) < 0.01 && 
					Math.abs(z - prev_z) < 0.01 && 
					Math.abs(yaw - prev_yaw) < 0.01 &&
					Math.abs(pitch - prev_pitch) < 0.01) {
					//System.out.println("Update complete: " + iter);
					break;
				}
				if (iter > 15) {
					break;
				}
			}
		}
	}
}
