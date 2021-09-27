import torch
class MB_alg:

    def __init__(self,est_model,actor,t_step,n_parametrised_steps, velocity_w, th_error, n_arms):

        self.est_model = est_model
        self.actor = actor
        self.n_parametrised_steps = n_parametrised_steps
        self.velocity_weight = velocity_w
        self.n_arms = n_arms

        self.t_step = t_step

        self.ModelError_th = 0.00001#0.0001#0.015
        self.th_error = th_error

        self.max_model_ep = 1000


    def update_model(self, actions, trg_y, target_arm):



        trg_y = trg_y.squeeze()
        est_y = self.est_model.perform_reaching(self.t_step,actions).squeeze()
        ep = 0


        ep_acc = []
        ep_loss = []


        max_error = torch.mean(torch.max(torch.abs(trg_y - est_y),dim=0)[0])

        print("1° max error: ", max_error,"\n")


        while ep < self.max_model_ep and max_error > self.ModelError_th:

           model_loss = self.est_model.update(trg_y, est_y)

           est_y = self.est_model.perform_reaching(self.t_step,actions).squeeze()

           ep_loss.append(torch.mean(model_loss.detach()))
           max_error = torch.mean(torch.max(torch.abs(trg_y - est_y),dim=0)[0])

           ep_acc.append(max_error.detach())
           ep += 1


           if ep % 50 == 0:

               avr_error = sum(ep_acc) /50
               avr_loss = sum(ep_loss)/50

               #self.est_model.ln_decay()

               print("Model upd ep: ",ep)
               print("avr max error: ", avr_error)
               print("avr loss: ", avr_loss)

               ep_acc = []
               ep_loss = []


        return ep


    def update_actor(self, target_states,f_points):

         ep = 0
         ep_distance = []
         #ep_velocity = []
         print_rwd = 50 # initialise to high random value

         while print_rwd > self.th_error:

                ep += 1

                actions = self.actor(target_states).view(-1,2,self.n_parametrised_steps)


                thetas = self.est_model.perform_reaching(self.t_step, actions)

                rwd = self.est_model.multiP_compute_rwd(thetas,target_states[:,0:1],target_states[:,1:2], f_points, self.n_arms)
                velocity = self.est_model.compute_vel(thetas, -1)
                loss = rwd + (velocity * self.velocity_weight)

                self.actor.update(loss)

                ep_distance.append(torch.mean(torch.sqrt(rwd).detach()))  # mean distance to assess performance
                #ep_velocity.append(torch.sqrt(velocity).detach())


                if ep % 1 == 0:

                    print_rwd = sum(ep_distance) / 1

                    print("Actor update ep: ", ep)
                    print("Accuracy: ", print_rwd)
                    ep_distance = []
                    #ep_velocity = []

         return ep